import os
from functools import reduce

import tensorflow as tf


class SingleNetwork(tf.keras.Model):
    def __init__(self, layer_list, model_name, checkpoint_dir='temp/td3'):
        super(SingleNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_td3')
        self.layer_list = layer_list

    def call(self, state):
        return reduce(lambda input_data, l: l(input_data), self.layer_list, state)
