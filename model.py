import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')
GPUs = tf.config.experimental.list_physical_devices('GPU')

if GPUs:
    try:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Model(tf.keras.Model):
    def __init__(self, num_actions, state_space, parameters):
        """Use Model Parameters from torch-rl repo (https://github.com/lcswillems/rl-starter-files)"""
        super().__init__(Model)
        tf.random.set_seed(parameters['seed'])
        num_units = parameters['num_units']
        n = state_space.shape[0]
        m = state_space.shape[1]
        self.image_embedding_size = (n - 5) * (m - 5) * 64
        self.convolution_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.Flatten()
        ])
        self.actor = tf.keras.Sequential([self.convolution_layer,
                                          tf.keras.layers.Dense(num_units, activation='tanh'),
                                          tf.keras.layers.Dense(num_actions)
                                          ])
        self.critic = tf.keras.Sequential([self.convolution_layer,
                                           tf.keras.layers.Dense(num_units, activation='tanh'),
                                           tf.keras.layers.Dense(1)
                                           ])
        self.next_representation = tf.keras.Sequential([self.convolution_layer,
                                                        tf.keras.layers.Dense(num_units, activation='tanh'),
                                                        tf.keras.layers.Dense(self.image_embedding_size)
                                                        ])

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        input_tensor = tf.convert_to_tensor(inputs)
        next_rep = self.next_representation(inputs)
        return self.actor(input_tensor), self.critic(input_tensor)

    def representation(self, inputs):
        # Inputs is a numpy array, convert to a tensor.
        input_tensor = tf.convert_to_tensor(inputs)
        # Convolution Layer to Inputs
        latent_representation = self.convolution_layer(input_tensor)
        return latent_representation

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        # Sample actions
        action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
