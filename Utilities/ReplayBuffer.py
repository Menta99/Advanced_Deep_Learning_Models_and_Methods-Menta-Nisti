import numpy as np
import tensorflow as tf
from Utilities.SegmentTree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, size, shape, actions, alpha):
        assert size > 0 and size & (size - 1) == 0, 'size must me a positive power of 2'
        self.size = size

        assert shape is not None, 'shape must not be None'
        self.shape = shape

        assert actions > 0, 'actions must me greater than 0'
        self.actions = actions

        assert 0 <= alpha <= 1, 'alpha must me between 0 and 1 (0: no prioritization, 1: full prioritization)'
        self.alpha = alpha
        self.max_priority = 1.0
        self.epsilon = 0.01
        self.tree_sum = SumSegmentTree(self.size)
        self.tree_min = MinSegmentTree(self.size)

        self.memory_initial_state = np.zeros((self.size, *self.shape))
        self.memory_action = np.zeros((self.size, self.actions))
        self.memory_reward = np.zeros(self.size)
        self.memory_final_state = np.zeros((self.size, *self.shape))
        self.memory_terminal = np.zeros(self.size, dtype=bool)

        self.counter = 0

    def push(self, initial_state, action, reward, final_state, terminal):
        index = self.counter % self.size

        self.memory_initial_state[index] = initial_state
        self.memory_action[index] = action
        self.memory_reward[index] = reward
        self.memory_final_state[index] = final_state
        self.memory_terminal[index] = terminal

        self.counter += 1

        self.tree_sum[index] = self.max_priority ** self.alpha
        self.tree_min[index] = self.max_priority ** self.alpha

    def pop(self, batch_size, beta):
        assert 0 <= beta <= 1, 'beta must me between 0 and 1 (0: no correction, 1: full correction)'
        indexes = self._sample_proportional(batch_size)
        weights = []
        sum_value = self.tree_sum.sum()
        min_value = self.tree_min.min()
        min_prob = min_value / sum_value
        max_weight = (min_prob * self.size) ** (- beta)
        sample_prob = [self.tree_sum[i] / sum_value for i in indexes]
        weights = tf.convert_to_tensor([(p * self.size) ** (- beta) / max_weight for p in sample_prob],
                                       dtype=tf.float32)
        indexes = tf.convert_to_tensor(indexes, dtype=tf.int32)

        initial_states = tf.convert_to_tensor(self.memory_initial_state[indexes], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.memory_action[indexes], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.memory_reward[indexes], dtype=tf.float32)
        final_states = tf.convert_to_tensor(self.memory_final_state[indexes], dtype=tf.float32)
        terminals = self.memory_terminal[indexes]

        return initial_states, actions, rewards, final_states, terminals, weights, indexes

    def _sample_proportional(self, batch_size):
        total = self.tree_sum.sum(0, self.size - 1)
        mass = np.random.random(size=batch_size) * total
        indexes = [self.tree_sum.retrieve(elem) for elem in mass]
        return indexes

    def update_priorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(indexes) >= 0
        assert np.max(indexes) < self.size
        for i in range(len(indexes)):
            self.tree_sum[indexes[i]] = priorities[i] ** self.alpha
            self.tree_min[indexes[i]] = priorities[i] ** self.alpha

        self.max_priority = max(self.max_priority, np.max(priorities))

    def update_priorities_variant(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        assert np.min(priorities) >= 0
        assert np.min(indexes) >= 0
        assert np.max(indexes) < self.size
        priorities = np.minimum(priorities + self.epsilon, self.max_priority) ** self.alpha
        for i in range(len(indexes)):
            self.tree_sum[indexes[i]] = priorities[i]
            self.tree_min[indexes[i]] = priorities[i]
