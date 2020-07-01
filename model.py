import numpy as np
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, num_actions, state_space, parameters):
        """Use Model Parameters from torch-rl repo (https://github.com/lcswillems/rl-starter-files)"""
        super().__init__(Model)
        num_units = parameters['num_units']
        n = state_space.shape[0]
        m = state_space.shape[1]
        self.image_embedding_size = (n - 5) * (m - 5) * 64
        """Individual Layers"""
        self.representation = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.Flatten()
        ])

        self.actor_dense = tf.keras.layers.Dense(num_units, activation='tanh')
        self.actor_output = tf.keras.layers.Dense(num_actions)

        self.actor = tf.keras.Sequential([self.representation, self.actor_dense, self.actor_output])

        self.critic_dense = tf.keras.layers.Dense(num_units, activation='tanh')
        self.critic_output = tf.keras.layers.Dense(1)

        self.critic = tf.keras.Sequential([self.representation, self.critic_output])

        self.critic_model = tf.keras.Sequential([self.critic_output])

        self.h_tp1_dense = tf.keras.layers.Dense(num_units, activation='tanh')
        self.h_tp1_output = tf.keras.layers.Dense(self.image_embedding_size)

        self.h_tp1 = tf.keras.Sequential([self.h_tp1_dense, self.h_tp1_output])

        self.r_t_dense = tf.keras.layers.Dense(num_units, activation='tanh')
        self.r_t_output = tf.keras.layers.Dense(1)

        self.r_t = tf.keras.Sequential([self.r_t_dense, self.r_t_output])

    def call(self, inputs, **kwargs):
        input_tensor = tf.convert_to_tensor(inputs)
        latent_rep = self.representation(input_tensor)

        return self.actor_output(self.actor_dense(latent_rep)), self.critic_output(latent_rep)

    def predict_h_tp1(self, inputs):
        input_tensor = tf.convert_to_tensor(inputs)
        return self.h_tp1_output(self.h_tp1_dense(input_tensor))

    def predict_r_t(self, inputs):
        input_tensor = tf.convert_to_tensor(inputs)
        return self.r_t_output(self.r_t_dense(input_tensor))

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        # Sample actions
        action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def action_from_representation(self, representation):
        representation = tf.convert_to_tensor(representation)
        logits = self.actor_output(self.actor_dense(representation))
        action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1)
