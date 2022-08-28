import json
import traceback

import tensorflow as tf


class NetworkBuilder:
    def __init__(self):
        self.supported_layers = {'Input': tf.keras.layers.Input,
                                 'Dense': tf.keras.layers.Dense,
                                 'Conv2D': tf.keras.layers.Conv2D,
                                 'Activation': tf.keras.layers.Activation,
                                 'BatchNormalization': tf.keras.layers.BatchNormalization,
                                 'Dropout': tf.keras.layers.Dropout,
                                 'Flatten': tf.keras.layers.Flatten,
                                 'Concatenate': tf.keras.layers.Concatenate,
                                 'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                                 'GlobalMaxPooling2D': tf.keras.layers.GlobalMaxPooling2D}

    def build_network(self, params_dict):
        """
        Use the following format:
        dict =
            {0:
                {'name': 'LAYER_NAME_0',
                'params':
                    {'PARAM_NAME_0_0': PARAM_VALUE_0_0,
                    ...
                    'PARAM_NAME_0_N':   PARAM_VALUE_0_N}
                    }
            ...
            {N:
                {'name': 'LAYER_NAME_N',
                'params':
                    {'PARAM_NAME_N_0': PARAM_VALUE_N_0,
                    ...
                    'PARAM_NAME_N_N': PARAM_VALUE_N_N}
                    }
                }
            }
        :param params_dict: dictionary of network layers information
        :return: the sorted list of layers based on the params_dict
        """
        return [self.build_layer(layer_dict) for layer_dict in params_dict.values()]

    def build_network_from_file(self, filepath):
        with open(filepath) as f:
            data = f.read()
        return self.build_network(json.loads(data))

    def build_layer(self, layer_dict):
        """
        :param layer_dict: dictionary of layer information
        :return: layer build with passed parameters
        """
        try:
            return self.supported_layers[layer_dict['name']](**layer_dict['params'])
        except KeyError:
            print(traceback.format_exc())
            print("Use one of the supported layers {}".format(list(self.supported_layers.keys())))
        except TypeError:
            print(traceback.format_exc())
            print("Use valid parameters")
