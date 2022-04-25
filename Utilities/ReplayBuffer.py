import numpy as np

from Utilities.SegmentTree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, size, shape, actions, alpha):
        assert size > 0 and size & (size - 1) == 0, 'size must me a positive power of 2'
        self.size = size

        assert shape > 0, 'shape must me greater than 0'
        self.shape = shape

        assert actions > 0, 'actions must me greater than 0'
        self.actions = actions

        assert 0 <= alpha <= 1, 'alpha must me between 0 and 1 (0: no prioritization, 1: full prioritization)'
        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_sum = SumSegmentTree(self.size)
        self.tree_min = MinSegmentTree(self.size)

        self.memory_initial_state = np.zeros((self.size, *self.shape))
        self.memory_action = np.zeros((self.size, self.actions))
        self.memory_reward = np.zeros(self.size)
        self.memory_final_state = np.zeros((self.size, *self.shape))
        self.memory_terminal = np.zeros(self.size, dtype=np.bool)

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
        min_prob = self.tree_min.min() / self.tree_sum.sum()
        max_weight = (min_prob * self.size) ** (- beta)
        sample_prob = self.tree_sum[indexes] / self.tree_sum.sum()
        weights = (sample_prob * self.size) ** (- beta) / max_weight

        initial_states = self.memory_initial_state[indexes]
        actions = self.memory_action[indexes]
        rewards = self.memory_reward[indexes]
        final_states = self.memory_final_state[indexes]
        terminals = self.memory_terminal[indexes]

        return initial_states, actions, rewards, final_states, terminals, weights, indexes

    def _sample_proportional(self, batch_size):
        total = self.tree_sum.sum(0, self.size - 1)
        mass = np.random.random(size=batch_size) * total
        index = self.tree_sum.retrieve(mass)
        return index

    def update_priorities(self, indexes, priorities):
        assert len(indexes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(indexes) >= 0
        assert np.max(indexes) < self.size
        self.tree_sum[indexes] = priorities ** self.alpha
        self.tree_min[indexes] = priorities ** self.alpha

        self.max_priority = max(self.max_priority, np.max(priorities))
