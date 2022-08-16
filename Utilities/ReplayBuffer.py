import numpy as np
import tensorflow as tf
from Utilities.SegmentTree import SumSegmentTree, MinSegmentTree
import gym


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space):
        assert buffer_size > 0 and buffer_size & (buffer_size - 1) == 0, 'size must me a positive power of 2'
        self.buffer_size = buffer_size

        assert isinstance(observation_space, gym.spaces.Space), 'observation_space must be a space not of type {}'.format(type(observation_space))
        self.observation_space = observation_space

        assert isinstance(action_space, gym.spaces.Space), 'action_space must be a space not of type {}'.format(type(action_space))
        self.action_space = action_space

        self.observation_space_shape = self.observation_space.shape
        if type(self.action_space) == gym.spaces.Discrete:
            self.action_space_shape = (self.action_space.n,)
            self.action_dim = 1
        else:
            self.action_space_shape = self.action_space.nvec
            self.action_dim = self.action_space.shape[0]

        self.memory_initial_states = np.zeros((self.buffer_size, *self.observation_space_shape),
                                              dtype=observation_space.dtype)
        self.memory_actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.memory_rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.memory_final_states = np.zeros((self.buffer_size, *self.observation_space_shape),
                                            dtype=observation_space.dtype)
        self.memory_terminals = np.zeros(self.buffer_size, dtype=np.float32)

        self.counter = 0
        self.full = False

    def push(self, initial_state, action, reward, final_state, terminal):
        self.memory_initial_states[self.counter, :] = np.array(initial_state).copy()
        self.memory_actions[self.counter, :] = np.array(action).copy()
        self.memory_rewards[self.counter] = np.array(reward).copy()
        self.memory_final_states[self.counter, :] = np.array(final_state).copy()
        self.memory_terminals[self.counter] = np.array(terminal).copy()

        self.update_counter()

    def update_counter(self):
        self.counter += 1
        if self.counter == self.buffer_size:
            self.full = True
            self.counter = 0

    def pop(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.counter
        indexes = np.random.randint(0, upper_bound, size=batch_size)

        initial_states = tf.convert_to_tensor(self.memory_initial_states[indexes, :], dtype=np.float32)
        actions = tf.convert_to_tensor(self.memory_actions[indexes, :], dtype=np.float32)
        rewards = tf.convert_to_tensor(self.memory_rewards[indexes], dtype=np.float32)
        final_states = tf.convert_to_tensor(self.memory_final_states[indexes, :], dtype=np.float32)
        terminals = tf.convert_to_tensor(self.memory_terminals[indexes], dtype=np.float32)

        return initial_states, actions, rewards, final_states, terminals, None, indexes


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, alpha):
        super().__init__(buffer_size, observation_space, action_space)

        assert 0 <= alpha <= 1, 'alpha must me between 0 and 1 (0: no prioritization, 1: full prioritization)'
        self.alpha = alpha
        self.max_priority = 1.0
        self.epsilon = 0.01
        self.tree_sum = SumSegmentTree(self.buffer_size)
        self.tree_min = MinSegmentTree(self.buffer_size)

    def push(self, initial_state, action, reward, final_state, terminal):
        self.tree_sum[self.counter] = self.max_priority ** self.alpha
        self.tree_min[self.counter] = self.max_priority ** self.alpha

        super(PrioritizedReplayBuffer, self).push(initial_state, action, reward, final_state, terminal)

    def pop(self, batch_size, beta=0):
        assert 0 <= beta <= 1, 'beta must me between 0 and 1 (0: no correction, 1: full correction)'
        indexes = self._sample_proportional(batch_size)
        sum_value = self.tree_sum.sum()
        min_value = self.tree_min.min()
        min_prob = min_value / sum_value
        max_weight = (min_prob * self.buffer_size) ** (- beta)
        sample_prob = [self.tree_sum[i] / sum_value for i in indexes]
        weights = tf.expand_dims(tf.convert_to_tensor([(p * self.buffer_size) ** (- beta) /
                                                       max_weight for p in sample_prob], dtype=np.float32), axis=-1)

        initial_states = tf.convert_to_tensor(self.memory_initial_states[indexes, :], dtype=np.float32)
        actions = tf.convert_to_tensor(self.memory_actions[indexes, :], dtype=np.float32)
        rewards = tf.convert_to_tensor(self.memory_rewards[indexes], dtype=np.float32)
        final_states = tf.convert_to_tensor(self.memory_final_states[indexes, :], dtype=np.float32)
        terminals = tf.convert_to_tensor(self.memory_terminals[indexes], dtype=np.float32)

        return initial_states, actions, rewards, final_states, terminals, weights, indexes

    def _sample_proportional(self, batch_size):
        total = self.tree_sum.sum(0, self.buffer_size - 1)
        mass = np.random.random(size=batch_size) * total
        indexes = np.array([self.tree_sum.retrieve(elem) for elem in mass])
        return indexes

    def update_priorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        assert np.min(priorities) >= 0
        assert ((indexes >= 0) & (indexes < self.buffer_size)).all()
        priorities = np.minimum(priorities + self.epsilon, self.max_priority) ** self.alpha
        for i in range(len(indexes)):
            self.tree_sum[indexes[i]] = priorities[i]
            self.tree_min[indexes[i]] = priorities[i]
