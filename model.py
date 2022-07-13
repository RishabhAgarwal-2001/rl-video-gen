import numpy as np
import tensorflow as tf
from random import randint, sample

from tensorflow.keras.models import Model

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Reshape, BatchNormalization
from collections import deque

class Environment:
    def __init__(self, dataset_videos, dataset_labels):

        self.dataset = (dataset_videos, dataset_labels)
        self.number_of_videos = self.dataset[0].shape[0]
        self.number_of_frames = self.dataset[0].shape[1]
        self.frame_shape = self.dataset[0][0][0].shape
        self.reset()

    def get_complete_current_state(self):
        return self.current_state, self.current_label

    def reset(self):
        self.current_expected_state = (
            randint(0, self.number_of_videos-1), 0)  # (Video, Frame)
        self.current_state = self.dataset[0][self.current_expected_state[0]
                                             ][self.current_expected_state[1]]
        self.current_label = self.dataset[1][self.current_expected_state[0]]
        self.terminated = False

    def act(self, action):
        if self.terminated:
            return
        action = action - 1
        self.current_expected_state = (
            self.current_expected_state[0], self.current_expected_state[1] + 1)
        if self.current_expected_state[1] == self.number_of_frames - 1:
            self.terminated = True
        self.current_state = self.current_state + action
        rwrd = self.reward()
        self.current_state = np.clip(self.current_state, 0, 1)
        return rwrd, self.current_state, self.terminated, self.dataset[0][self.current_expected_state[0]][self.current_expected_state[1]]

    def reward(self):
        video_idx, frame_idx = self.current_expected_state
        r = -np.abs(np.subtract(self.current_state,
                    self.dataset[0][video_idx][frame_idx]))
        for row in range(self.frame_shape[0]):
            for col in range(self.frame_shape[1]):
                if r[row][col] == 0:
                    r[row][col] = 1
                elif r[row][col] == -1:
                    r[row][col] = -1
        return r

    def sample(self):
        return np.random.randint(3, size=self.frame_shape)


class Agent:
    def __init__(self, environment, optimizer, latent_dims):
        # Initialize Attributes
        self._state_size = environment.frame_shape
        self._latent_dims = latent_dims
        self._action_types = 3  # Walking, Jumping & Raising
        self._action_size = 3  # -1, 0, 1
        self.optimizer = optimizer
        self.environment = environment

        self.experience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def store(self, flatten_image, action, reward, next_state, terminated, multiplier=1):
        self.experience_replay.append(
            (flatten_image, action, reward, next_state, terminated, multiplier))

    def _build_compile_model(self):
        image_input_layer = Input(
            shape=(self._state_size[0], self._state_size[1], 2))
        conv_layer_1 = Conv2D(filters=16, kernel_size=3, padding="same",
                              activation=tf.keras.layers.LeakyReLU(alpha=0.1))(image_input_layer)
        batch_norm_1 = BatchNormalization()(conv_layer_1)
        conv_layer_2 = Conv2D(filters=32, kernel_size=5, padding="same",
                              activation=tf.keras.layers.LeakyReLU(alpha=0.1))(batch_norm_1)
        batch_norm_2 = BatchNormalization()(conv_layer_2)
        conv_layer_3 = Conv2D(filters=64, kernel_size=3, padding="same",
                              activation=tf.keras.layers.LeakyReLU(alpha=0.1))(batch_norm_2)
        batch_norm_3 = BatchNormalization()(conv_layer_3)
        output_layer = Conv2D(filters=self._action_size, kernel_size=3,
                              padding="same", activation='linear')(batch_norm_3)
        model = Model(inputs=image_input_layer, outputs=output_layer)
        model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer=self.optimizer)
        # model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state, act_randomly):

        state_frame = state[0]
        condition_matrix = state[1]

        reshaped_image = np.reshape(
            state_frame, (state_frame.shape[0], state_frame.shape[1], 1))
        reshaped_condition_matrix = np.reshape(
            condition_matrix, (state_frame.shape[0], state_frame.shape[1], 1))
        concat_image = np.concatenate(
            [reshaped_image, reshaped_condition_matrix], axis=2)
        concat_image = np.reshape(
            concat_image, (1, state_frame.shape[0], state_frame.shape[1], 2))

        if act_randomly:
            return self.environment.sample(), concat_image, None

        q_values = self.q_network.predict(concat_image)

        return np.argmax(q_values[0], axis=2), concat_image, q_values[0]

    def retrain(self, batch_size):
        minibatch = sample(self.experience_replay, batch_size)
        states = []

        for concat_image, action, reward, next_state, terminated, _ in minibatch:
            states.append(concat_image[0])

        states = np.asarray(states)
        targets = self.q_network.predict(states)

        i = 0
        for _, action, reward, next_state, terminated, condition_matrix in minibatch:
            reshaped_image = np.reshape(
                next_state, (next_state.shape[0], next_state.shape[1], 1))
            reshaped_condition_matrix = np.reshape(
                condition_matrix, (next_state.shape[0], next_state.shape[1], 1))
            concat_image = np.concatenate(
                [reshaped_image, reshaped_condition_matrix], axis=2)
            next_state = np.reshape(
                concat_image, (1, next_state.shape[0], next_state.shape[1], 2))

            if not terminated:
                t = self.target_network.predict(next_state)

            for row in range(self._state_size[0]):
                for col in range(self._state_size[1]):
                    if terminated:
                        targets[i][row][col][action[row]
                                             [col]] = reward[row][col]
                    else:
                        targets[i][row][col][action[row][col]] = reward[row][col] + \
                            self.gamma * np.amax(t[0][row][col])
            i += 1

        self.q_network.fit(states, targets, epochs=256,
                           verbose=0, batch_size=16)

    def save(self, path):
        self.q_network.save(path + f'model')

    def load_model(self, path):
        self.q_network = tf.keras.models.load_model(path + f'model')
        self.target_network = tf.keras.models.load_model(path + f'model')
