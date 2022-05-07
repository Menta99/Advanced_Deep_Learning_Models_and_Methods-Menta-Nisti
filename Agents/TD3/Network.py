import tensorflow as tf
import os
from functools import reduce


class CriticNetwork(tf.keras.Model):
    def __init__(self, layer_list_ext, layer_list_head, model_name, checkpoint_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_td3')
        self.layer_list_ext = layer_list_ext
        self.layer_list_head = layer_list_head

    def call(self, state, action):
        flatten_out = reduce(lambda input_data, l: l(input_data), self.layer_list_ext, state)
        return reduce(lambda input_data, l: l(input_data), self.layer_list_head,
                      tf.concat([flatten_out, *action], axis=-1))


class ActorNetwork(tf.keras.Model):
    def __init__(self, layer_list_base, layer_list_heads, model_name, checkpoint_dir='temp/td3'):
        super(ActorNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_td3')
        self.layer_list_base = layer_list_base
        self.layer_list_heads = layer_list_heads

    def call(self, state):
        base_out = reduce(lambda input_data, l: l(input_data), self.layer_list_base, state)
        return tuple(layer(base_out) for layer in self.layer_list_heads)
