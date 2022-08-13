import gym
import numpy as np
import tensorflow as tf
from typing import NamedTuple


class ReplayBufferSamples(NamedTuple):
    observations: tf.Tensor
    actions: tf.Tensor
    next_observations: tf.Tensor
    dones: tf.Tensor
    rewards: tf.Tensor


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = self.observation_space.shape
        if type(self.action_space) == gym.spaces.Discrete:
            self.action_space_shape = (self.action_space.n,)
            self.action_dim = 1
        else:
            self.action_space_shape = self.action_space.nvec
            self.action_dim = self.action_space.shape[0]

        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        data = (self.observations[batch_inds, :], self.actions[batch_inds, :],
                self.next_observations[batch_inds, :], self.dones[batch_inds],
                self.rewards[batch_inds])
        return ReplayBufferSamples(*tuple(map(to_tensor, data)))


def to_tensor(array):
    return tf.convert_to_tensor(array)
