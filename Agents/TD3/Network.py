import tensorflow as tf
import os
from functools import reduce


class CriticNetwork(tf.keras.Model):
    def __init__(self, layers, model_name, checkpoint_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_td3')
        self.layers = layers

    def call(self, state, action):
        return reduce(lambda input_data, l: l(input_data), self.layers, tf.concat([state, action], axis=1))


class ActorNetwork(tf.keras.Model):
    def __init__(self, layers, model_name, checkpoint_dir='temp/td3'):
        super(ActorNetwork, self).__init__()
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_td3')
        self.layers = layers

    def call(self, state):
        return reduce(lambda input_data, l: l(input_data), self.layers, state)
