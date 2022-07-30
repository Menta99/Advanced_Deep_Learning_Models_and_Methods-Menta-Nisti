import itertools
import os

import tensorflow as tf
from tensorflow import keras

from Agents.DDDQN.DDDQNAgent import DDDQNAgent
from Utilities.ConnectFour import ConnectFourEnv
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.TrainWizard import TurnGameTrainWizard
from Utilities.Wrappers import OpponentWrapper


def get_turn(config):
    if config[3] == 'First':
        return True
    elif config[3] == 'Second':
        return False
    elif config[3] == 'Random':
        return None
    else:
        raise ValueError('Turn provided does not exist!')


def get_env(config, turn):
    if config[0] == 'TicTacToe':
        return OpponentWrapper(TicTacToeEnv(representation, turn), opponent)
    elif config[0] == 'ConnectFour':
        return OpponentWrapper(ConnectFourEnv(representation, turn), opponent)
    else:
        raise ValueError('Game provided does not exist!')


def get_network_dicts(representation):
    if representation == 'Tabular':
        network_dict_base = {0:
                                 {'name': 'Dense',
                                  'params': {
                                      'units': 64,
                                      'activation': 'relu',
                                      'kernel_initializer': tf.keras.initializers.HeNormal()
                                  }},
                             1:
                                 {'name': 'Dense',
                                  'params': {
                                      'units': 32,
                                      'activation': 'relu',
                                      'kernel_initializer': tf.keras.initializers.HeNormal()
                                  }}}
    else:
        network_dict_base = {0:
                                 {'name': 'Conv2D',
                                  'params': {
                                      'filters': 64,
                                      'kernel_size': (8, 8),
                                      'strides': (4, 4),
                                      'activation': 'relu',
                                      'kernel_initializer': tf.keras.initializers.HeNormal()
                                  }},
                             1:
                                 {'name': 'Conv2D',
                                  'params': {
                                      'filters': 32,
                                      'kernel_size': (4, 4),
                                      'strides': (2, 2),
                                      'activation': 'relu',
                                      'kernel_initializer': tf.keras.initializers.HeNormal()
                                  }},
                             2:
                                 {'name': 'Conv2D',
                                  'params': {
                                      'filters': 16,
                                      'kernel_size': (3, 3),
                                      'activation': 'relu',
                                      'kernel_initializer': tf.keras.initializers.HeNormal()
                                  }}}
    network_dict_advantage = {2:
                                  {'name': 'Flatten',
                                   'params': {}
                                   },
                              3:
                                  {'name': 'Dense',
                                   'params': {
                                       'units': env.action_space.n,
                                       'activation': 'relu',
                                       'kernel_initializer': tf.keras.initializers.HeNormal()
                                   }}}
    network_dict_value = {4:
                              {'name': 'Flatten',
                               'params': {}
                               },
                          5:
                              {'name': 'Dense',
                               'params': {
                                   'units': 1,
                                   'activation': 'relu',
                                   'kernel_initializer': tf.keras.initializers.HeNormal()
                               }}}
    return network_dict_base, network_dict_advantage, network_dict_value


def get_agent(env, network_dict_base, network_dict_advantage, network_dict_value, network_path):
    return DDDQNAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      q_net_dict=[network_dict_base, network_dict_advantage, network_dict_value],
                      q_target_net_dict=[network_dict_base, network_dict_advantage, network_dict_value],
                      double_q=True,
                      dueling_q=True,
                      q_net_update=4,
                      q_target_net_update=10000,
                      discount_factor=0.99,
                      q_net_optimizer=keras.optimizers.Adam,
                      q_target_net_optimizer=keras.optimizers.Adam,
                      q_net_learning_rate=1e-5,
                      q_target_net_learning_rate=1e-5,
                      q_net_loss=keras.losses.Huber(),
                      q_target_net_loss=keras.losses.Huber(),
                      num_episodes=100000,
                      memory_size=8192,
                      memory_alpha=0.7,
                      memory_beta=0.4,
                      max_epsilon=1.0,
                      min_epsilon=0.001,
                      epsilon_A=0.35,
                      epsilon_B=0.25,
                      epsilon_C=0.1,
                      batch_size=32,
                      checkpoint_dir=network_path)


if __name__ == '__main__':
    for config in itertools.product(*[['TicTacToe', 'ConnectFour'], ['Tabular', 'Graphic'], ['Random', 'MinMax'],
                                      ['First', 'Second', 'Random']]):
        print('Executing the following config: {}'.format(config))
        algorithm = 'DDDQN'
        environment = config[0]
        representation = config[1]
        opponent = config[2]
        agent_turn = config[3]

        config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
        data_path = '..\\Results\\' + config_name + '\\'
        gif_path = data_path + 'GIFs\\'
        network_path = data_path + 'NetworkParameters\\'
        os.mkdir(data_path)
        os.mkdir(gif_path)
        os.mkdir(network_path)

        turn = get_turn(config)
        env = get_env(config, turn)
        network_dict_base, network_dict_advantage, network_dict_value = get_network_dicts(representation)
        agent = get_agent(env, network_dict_base, network_dict_advantage, network_dict_value, network_path)

        wizard = TurnGameTrainWizard(environment=environment,
                                     agent=agent,
                                     objective_score=1,
                                     running_average_length=100,
                                     evaluation_steps=1000,
                                     evaluation_games=100,
                                     representation=representation,
                                     agent_turn=turn,
                                     agent_turn_test=None,
                                     opponent=opponent,
                                     path=data_path)

        wizard.train()
