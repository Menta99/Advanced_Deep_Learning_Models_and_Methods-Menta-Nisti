import tensorflow as tf
import os
from functools import reduce


class DuelingNetwork(tf.keras.Model):
    def __init__(self, layer_list_base, layer_list_advantage, layer_list_value, model_name, checkpoint_dir='tmp/dddqn'):
        super(DuelingNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_dddqn')
        self.layer_list_base = layer_list_base
        self.layer_list_advantage = layer_list_advantage
        self.layer_list_value = layer_list_value

    def call(self, state):
        state = reduce(lambda input_data, l: l(input_data), self.layer_list_base, state)
        a = reduce(lambda input_data, l: l(input_data), self.layer_list_advantage, state)
        v = reduce(lambda input_data, l: l(input_data), self.layer_list_value, state)
        return v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))

    def advantage(self, state):
        state = reduce(lambda input_data, l: l(input_data), self.layer_list_base, state)
        return reduce(lambda input_data, l: l(input_data), self.layer_list_advantage, state)


class DQNetwork(tf.keras.Model):
    def __init__(self, layer_list, model_name, checkpoint_dir='tmp/ddqn'):
        super(DQNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddqn')
        self.layer_list = layer_list

    def call(self, state):
        return reduce(lambda input_data, l: l(input_data), self.layer_list, state)
