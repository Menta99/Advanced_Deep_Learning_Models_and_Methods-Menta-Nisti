import operator
import numpy as np


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, 'capacity must me a positive power of 2'
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        self.memory = np.full(2 * self.capacity, self.neutral_element)

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.memory[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=0):
        if end < 0:
            end += self.capacity - 1
        return self._reduce_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.memory[idx] = val

        idx //= 2
        while idx >= 1:
            self.memory[idx] = self.operation(self.memory[2 * idx], self.memory[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        return self.memory[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=0):
        return super(SumSegmentTree, self).reduce(start, end)

    def retrieve(self, upperbound):
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)
        idx = 1

        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if self.memory[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.memory[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self, start=0, end=0):
        return super(MinSegmentTree, self).reduce(start, end)
