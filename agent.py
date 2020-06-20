import tensorflow as tf
import numpy as np
from collections import deque
import random
import datetime
from tqdm import tqdm


class A2C(tf.keras.Model):
    def __init__(self, model, n_output, parameters, loss_coefficients):
        """Main Algorithm Class"""
        super().__init__(A2C)
        np.random.seed(parameters['seed'])
        self.value_c = loss_coefficients['value']
        self.entropy_c = loss_coefficients['entropy']
        self.rep_c = loss_coefficients['representation']
        self.n_output = n_output
        self.batch_size = parameters['batch_size']
        self.gamma = parameters['gamma']
        self.gae_lambda = parameters['gae_lambda']
        self.model = model
        self.rep_size = self.model.image_embedding_size
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.RMSprop(lr=parameters['learning_rate'])
        self.critic_optimizer = tf.keras.optimizers.RMSprop(lr=parameters['learning_rate'])
        self.next_rep_optimizer = tf.keras.optimizers.RMSprop(lr=parameters['rep_learning_rate'])

    def act(self, state):
        action, _ = self.model.action_value(state)
        return action[0]

    def random_act(self, state):
        return np.random.choice(self.n_output)

    def value_loss(self, returns, value):
        # Mean Squared TD Error
        return self.value_c * tf.keras.losses.mean_squared_error(returns, value)

    def logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
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

    def representation_loss(self, actual_rep, predicted_rep):
        # Mean Squared Error
        return self.rep_c * tf.keras.losses.mean_squared_error(actual_rep, predicted_rep)

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
        observations = np.empty((self.batch_size,) + env.state_space.shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        losses = []
        next_obs = env.reset()
        for update in tqdm(range(updates)):
            # Make mini-batches for training
            for step in range(self.batch_size):
                # Get Observation (preprocessed)
                observations[step] = next_obs.copy()
                # Get action from Model
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                # Take Action
                next_obs, rewards[step], done[step], _ = env.step(actions[step])
                ep_rewards[-1] += rewards[step]
                if done[step]:  # end of episode
                    episode_count = len(ep_rewards)
                    # Print Result
                    if episode_count % logs['log_interval'] == 0:
                        reward_last_interval = np.mean(ep_rewards[-logs['log_interval']:])
                        print('\r Episode: {}, Episode Reward: {:.2f} \r'.format(episode_count, reward_last_interval))
                        if verbose:
                            with open(logs['log_file_name'], 'a') as f_episode:
                                f_episode.write('Episode: {}, Episode Reward: {:.2f} \n'.format(episode_count,
                                                                                                reward_last_interval))
                                f_episode.flush()
                    ep_rewards.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            observation_new = np.append(observations[1:, ], next_obs[None, :], axis=0)  # Batch of S_{t+1}
            target_rep = self.model.representation(observation_new)

            # returns, adv = self.returns_advantages(rewards, done, values, next_value)  # Normal Advantage Estimation
            returns, adv = self.gae_returns_advantages(rewards, done, values, next_value)  # GAE

            # Trick to input actions and advantages through same API.
            acts_and_adv = np.concatenate([actions[:, None], adv[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            loss = self.minibatch_train(observations, returns, acts_and_adv, observation_new)
            losses.append(loss)

            # Write Current Batch Results to Log File
            if len(losses) % logs['log_interval'] == 0:
                losses_interval = np.mean(losses[-logs['log_interval']:][0])
                if verbose:
                    with open(logs['log_file_name'], 'a') as f_batch:
                        f_batch.write('{} batches done: Loss: {:.5f}, Time: {} \n'.
                                      format(len(losses), losses_interval,
                                             datetime.datetime.now().strftime('%H:%M:%S')))
                        f_batch.flush()

        if verbose:
            with open(logs['log_file_name'], 'a') as f_close:
                f_close.write('Finished Training: {} \n'.format(datetime.datetime.now().strftime('%H:%M:%S')))
                f_close.close()
        return ep_rewards[:-1], losses

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

    def minibatch_train(self, observation, returns, acts_and_adv, observation_new):
        """Manual Training"""
        # Train Representation by learning to predict next_state
        with tf.GradientTape() as tape:
            next_hidden_state = self.model.next_representation(observation)
            """Allow gradient for target to also backpropogate. Use tf.stop_gradient() to stop this"""
            target_rep = self.model.representation(observation_new)
            representation_loss = tf.reduce_mean(self.representation_loss(target_rep, next_hidden_state))
        grads = tape.gradient(representation_loss, self.model.next_representation.trainable_variables)
        self.next_rep_optimizer.apply_gradients(zip(grads, self.model.next_representation.trainable_variables))

        # Train Actor
        with tf.GradientTape() as tape:
            logits = self.model.actor(observation)
            policy_loss = tf.reduce_mean(self.logits_loss(acts_and_adv, logits))
        grads = tape.gradient(policy_loss, self.model.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.model.actor.trainable_variables))

        # Train Critic
        with tf.GradientTape() as tape:
            value = self.model.critic(observation)
            value_loss = tf.reduce_mean(self.value_loss(returns, value))
        grads = tape.gradient(value_loss, self.model.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.model.critic.trainable_variables))

        return [representation_loss.numpy(), policy_loss.numpy(), value_loss.numpy()]

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


class Memory:
    """Experience Replay Implemented (Not used for A2C)"""

    def __init__(self, buffer_size, seed):
        self.buffer = deque(maxlen=buffer_size)
        random.seed(seed)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return [np.squeeze(state_batch), np.squeeze(action_batch), np.squeeze(reward_batch),
                np.squeeze(next_state_batch), np.squeeze(done_batch)]

    # Add observations, actions, rewards to memory
    def add_to_memory(self, data):
        self.buffer.append(data)
