import tensorflow as tf
import numpy as np
from collections import deque
import random
import datetime
from tqdm import tqdm
from model import Model

tf.keras.backend.set_floatx('float64')
GPUs = tf.config.experimental.list_physical_devices('GPU')

if GPUs:
    try:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class A2C(tf.keras.Model):
    def __init__(self, num_actions, state_space, arch_params, model_params, algorithm_params, loss_coefficients):
        """Main Algorithm Class"""
        super().__init__(A2C)
        np.random.seed(algorithm_params['seed'])
        tf.random.set_seed(algorithm_params['seed'])
        self.value_c = loss_coefficients['value']
        self.entropy_c = loss_coefficients['entropy']
        self.n_output = num_actions
        self.batch_size = algorithm_params['batch_size']
        self.gamma = algorithm_params['gamma']
        self.gae_lambda = algorithm_params['gae_lambda']
        self.state_space = state_space

        self.model = Model(num_actions, state_space, arch_params)
        self.rep_size = self.model.image_embedding_size
        self.buffer = Memory(model_params['buffer_size'], algorithm_params['seed'])
        self.use_model = model_params['use_model']
        self.planning_steps = model_params['planning_steps']
        self.rollouts = model_params['rollouts']

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.RMSprop(lr=algorithm_params['learning_rate'])
        self.critic_optimizer = tf.keras.optimizers.RMSprop(lr=algorithm_params['learning_rate'])
        self.next_rep_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['rep_learning_rate'])
        self.reward_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['reward_learning_rate'])
        self.model_critic_optimizer = tf.keras.optimizers.RMSprop(lr=model_params['model_critic_learning_rate'])

        # Model Losses
        self.model.actor.compile(optimizer=self.actor_optimizer, loss=self.logits_loss)
        self.model.critic.compile(optimizer=self.critic_optimizer, loss=self.value_loss)
        self.model.h_tp1.compile(optimizer=self.next_rep_optimizer, loss=self.mse_loss)
        self.model.r_t.compile(optimizer=self.reward_optimizer, loss=self.mse_loss)
        # self.model.critic.compile(optimizer=self.model_critic_optimizer, loss=self.value_loss)

    def act(self, state):
        action, _ = self.model.action_value(state)
        return action[0]

    def random_act(self, state):
        return np.random.choice(self.n_output)

    def value_loss(self, returns, value):
        # Mean Squared TD Error
        return self.value_c * tf.keras.losses.mean_squared_error(returns, value)

    def logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Policy loss is defined by policy gradients, weighted by advantages, (on actions taken)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # Entropy loss can be calculated as cross-entropy over itself.
        probabilities = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probabilities, probabilities)
        # Minimize policy and maximize entropy losses.
        return policy_loss - self.entropy_c * entropy_loss

    @staticmethod
    def mse_loss(actual, predicted):
        # Mean Squared Error
        return tf.keras.losses.mean_squared_error(actual, predicted)

    def train(self, env, frames, logs, verbose):
        # Start training
        if verbose:
            with open(logs['log_file_name'], 'a') as f_start:
                f_start.write('Starting Training: {} \n'.format(datetime.datetime.now().strftime('%H:%M:%S')))
                f_start.flush()
        # Number of Updates
        updates = int(frames // self.batch_size)
        # Storage helpers for a single batch of data.
        actions = np.empty((self.batch_size,), dtype=np.int32)
        rewards, done, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + self.state_space.shape)

        self.model_loss = []
        self.ep_rewards = [0.0]
        self.total_loss = []

        # Training loop: collect samples, send to optimizer, repeat updates times.
        next_obs = env.reset()
        for update in tqdm(range(updates)):
            # Store mini-batches for training
            for step in range(self.batch_size):
                # Get Observation (preprocessed)
                observations[step] = next_obs.copy()
                # Get action from Model
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                # Add to buffer
                if self.use_model:
                    self.buffer.add_to_memory((observations[step], actions[step]))
                # Take Action
                next_obs, rewards[step], done[step], _ = env.step(actions[step])
                self.ep_rewards[-1] += rewards[step]
                if done[step]:  # end of episode
                    episode_count = len(self.ep_rewards)
                    # Print Result
                    if episode_count % logs['log_interval'] == 0:
                        reward_last_interval = np.mean(self.ep_rewards[-logs['log_interval']:])
                        print('\r Episode: {}, Episode Reward: {:.2f} \r'.format(episode_count, reward_last_interval))
                        if verbose:
                            with open(logs['log_file_name'], 'a') as f_episode:
                                f_episode.write('Episode: {}, Episode Reward: {:.2f} \n'.format(episode_count,
                                                                                                reward_last_interval))
                                f_episode.flush()
                    self.ep_rewards.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])  # Get next state for last transition in batch
            # Train on stored mini-batch
            next_state_loss, reward_loss = self.train_next_representation_and_rewards(observations, actions, next_obs, rewards)
            policy_loss, value_loss = self.train_actor_and_critic(observations, rewards, done, values, next_value, actions)
            loss = [next_state_loss, reward_loss, policy_loss, value_loss]

            self.total_loss.append(loss)
            # Model based Critic Update
            if self.use_model:
                for _ in range(self.planning_steps):
                    self.model_based_update()

            # Write Current Batch Results to Log File
            if len(self.total_loss) % logs['log_interval'] == 0:
                losses_interval = np.mean(self.total_loss[-logs['log_interval']:][0])
                if verbose:
                    with open(logs['log_file_name'], 'a') as f_batch:
                        f_batch.write('{} batches done: Loss: {:.5f}, Time: {} \n'.
                                      format(len(self.total_loss), losses_interval,
                                             datetime.datetime.now().strftime('%H:%M:%S')))
                        f_batch.flush()

        if verbose:
            with open(logs['log_file_name'], 'a') as f_close:
                f_close.write('Finished Training: {} \n'.format(datetime.datetime.now().strftime('%H:%M:%S')))
                f_close.close()
        loss_history = {'loss': self.total_loss, 'model_loss': self.model_loss}
        return self.ep_rewards[:-1], loss_history

    def train_next_representation_and_rewards(self, observation, action, next_ob, reward):
        next_state_loss = 0
        reward_loss = 0
        if self.use_model:
            latent_representation = self.model.representation(observation)
            latent_representation_and_action = np.column_stack((latent_representation, action))
            observation_new = np.append(observation[1:, ], next_ob[None, :], axis=0)  # Batch of S_{t+1}
            target_rep = self.model.representation(observation_new).numpy()
            next_state_loss = self.model.h_tp1.train_on_batch(latent_representation_and_action, target_rep)
            reward_loss = self.model.h_tp1.train_on_batch(latent_representation_and_action, reward)
        return next_state_loss, reward_loss

    def train_actor_and_critic(self, observations, rewards, done, values, next_value, actions):
        # returns, adv = self.returns_advantages(rewards, done, values, next_value)  # Normal Advantage Estimation
        returns, adv = self.gae_returns_advantages(rewards, done, values, next_value)  # GAE
        acts_and_adv = np.concatenate([actions[:, None], adv[:, None]], axis=-1)

        # Performs a full training step on the collected batch.
        policy_loss = self.model.actor.train_on_batch(observations, acts_and_adv)
        value_loss = self.model.critic.train_on_batch(observations, returns)
        return policy_loss, value_loss

    def model_based_update(self):
        if self.use_model:
            state, action = self.buffer.sample()
            current_representation = self.model.representation(state[None, :])
            for _ in range(self.rollouts):
                current_representation_and_action = np.column_stack((current_representation, action))
                next_representation = self.model.h_tp1(current_representation_and_action)
                reward = self.model.r_t(current_representation_and_action)
                next_value = self.model.critic_model(next_representation)
                target_value = reward + self.gamma * next_value
                with tf.GradientTape() as tape:
                    value = self.model.critic_model(current_representation)
                    model_value_loss = tf.reduce_mean(self.value_loss(target_value, value))
                grads = tape.gradient(model_value_loss, self.model.critic_model.trainable_variables)
                self.model_critic_optimizer.apply_gradients(zip(grads, self.model.critic_model.trainable_variables))
                self.model_loss.append(model_value_loss.numpy())
                current_representation = next_representation
                action = self.model.action_from_representation(current_representation)

    def returns_advantages(self, rewards, done, values, next_value):
        """Using Normal Advantage Estimation"""
        # `last_value` is the last value estimate of the end of the mini-batch.
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - done[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages

    def gae_returns_advantages(self, rewards, done, values, last_value):
        """Using Generalized Advantage Estimation(GAE)"""
        # `last_value` is the last value estimate of the end of the mini-batch.
        steps = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        value = np.append(values, last_value)  # All values till t+1
        for t in reversed(range(steps)):
            # Delta = R(st) + gamma * V(t+1) * next_non_terminal  - V(st)
            delta = rewards[t] + self.gamma * value[t + 1] * (1 - done[t]) - value[t]
            # Advantage = delta + gamma *  λ (lambda) * next_non_terminal  * last_gae_lam
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - done[t]) * last_gae_lam

        # Returns are equal to advantages + baseline (value estimates)
        returns = advantages + values
        return returns, advantages

    def test_random(self, env, steps):
        """Take Random Actions in the Environment"""
        state = env.reset()
        episode_reward = 0
        episode_count = 0
        for step in tqdm(range(steps)):
            action = self.random_act(state)
            state, rewards, done, _ = env.step(action)
            episode_reward += rewards
            if done:
                episode_count += 1
                print('\r Episode: {}, Episode Reward: {} \r'.format(episode_count, episode_reward))
                episode_reward = 0
                state = env.reset()
        return episode_reward

    def test(self, env, steps):
        """Take Model Actions in the Environment"""
        state = env.reset()
        episode_reward = 0
        episode_count = 0
        for step in tqdm(range(steps)):
            action = self.act(state[None, :])
            state, rewards, done, _ = env.step(action)
            episode_reward += rewards
            if done:
                episode_count += 1
                print('\r Episode: {}, Episode Reward: {} \r'.format(episode_count, episode_reward))
                episode_reward = 0
                state = env.reset()
        return episode_reward

    def minibatch_train(self, observation, returns, acts_and_adv, observation_new, rewards):
        """Manual Training"""
        """Run thia in main loop:
        loss = self.minibatch_train(observations, returns, acts_and_adv, observation_new, rewards) """
        # Learn to predict the next state
        with tf.GradientTape() as tape:
            next_hidden_state = self.model.train_next_state(self.model.representation(observation))
            """Allow gradient for target to also backpropogate. Use tf.stop_gradient() to stop this"""
            target_rep = self.model.representation(observation_new)
            representation_loss = tf.reduce_mean(self.mse_loss(target_rep, next_hidden_state))
        grads = tape.gradient(representation_loss, self.model.train_next_state.trainable_variables)
        self.next_rep_optimizer.apply_gradients(zip(grads, self.model.train_next_state.trainable_variables))

        # Learn to predict the reward
        with tf.GradientTape() as tape:
            predicted_reward = self.model.train_reward(self.model.representation(observation))
            """Allow gradient for target to also backpropogate. Use tf.stop_gradient() to stop this"""
            target_reward = rewards
            reward_loss = tf.reduce_mean(self.mse_loss(target_reward, predicted_reward))
        grads = tape.gradient(reward_loss, self.model.train_reward.trainable_variables)
        self.next_reward_optimizer.apply_gradients(zip(grads, self.model.train_reward.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            logits = self.model.train_actor(observation)
            policy_loss = tf.reduce_mean(self.logits_loss(acts_and_adv, logits))
        grads = tape.gradient(policy_loss, self.model.train_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.model.train_actor.trainable_variables))

        # Train Critic
        with tf.GradientTape() as tape:
            value = self.model.train_critic(observation)
            value_loss = tf.reduce_mean(self.value_loss(returns, value))
        grads = tape.gradient(value_loss, self.model.train_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.model.train_critic.trainable_variables))

        return [representation_loss.numpy(), reward_loss.numpy(), policy_loss.numpy(), value_loss.numpy()]


class Memory:
    """Experience Replay Implemented (Not used for A2C)"""

    def __init__(self, buffer_size, seed):
        self.buffer = deque(maxlen=buffer_size)
        random.seed(seed)

    def __len__(self):
        return len(self.buffer)

    def sample(self):
        return np.squeeze(random.sample(self.buffer, 1))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return np.squeeze(batch)

    # Add observations, actions, rewards to memory
    def add_to_memory(self, data):
        self.buffer.append(data)
