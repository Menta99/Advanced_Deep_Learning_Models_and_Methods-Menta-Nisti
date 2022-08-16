import itertools
import os
import tensorflow as tf

from Agents.DDDQN.DDDQNAgent import DDDQNAgent
from TrainWizard import OpponentWrapper, TurnGameTrainWizard
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.ConnectFour import ConnectFourEnv


def get_turn(config):
    if config[3] == 'First':
        return True
    elif config[3] == 'Second':
        return False
    elif config[3] == 'Random':
        return None
    else:
        raise ValueError('Turn provided does not exist!')


def get_env(config, turn, representation, opponent):
    if config[0] == 'TicTacToe':
        return OpponentWrapper(TicTacToeEnv(representation, turn), opponent)
    elif config[0] == 'ConnectFour':
        return OpponentWrapper(ConnectFourEnv(representation, turn), opponent)
    else:
        raise ValueError('Game provided does not exist!')


def get_network_dicts(representation, env, dueling):
    if representation == 'Tabular':
        if dueling:
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
            network_dict_advantage = {2:
                                          {'name': 'Flatten',
                                           'params': {}
                                           },
                                      3:
                                          {'name': 'Dense',
                                           'params': {
                                               'units': 512,
                                               'activation': 'relu',
                                               'kernel_initializer': tf.keras.initializers.HeNormal()
                                           }},
                                      4:
                                          {'name': 'Dense',
                                           'params': {
                                               'units': env.action_space.n,
                                               'activation': None
                                           }}}
            network_dict_value = {5:
                                      {'name': 'Flatten',
                                       'params': {}
                                       },
                                  6:
                                      {'name': 'Dense',
                                       'params': {
                                           'units': 512,
                                           'activation': 'relu',
                                           'kernel_initializer': tf.keras.initializers.HeNormal()
                                       }},
                                  7:
                                      {'name': 'Dense',
                                       'params': {
                                           'units': 1,
                                           'activation': None
                                       }}}
        else:
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
                                      }},
                                 2:
                                     {'name': 'Flatten',
                                      'params': {}
                                      },
                                 3:
                                     {'name': 'Dense',
                                      'params': {
                                          'units': 64,
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 4:
                                     {'name': 'Dense',
                                      'params': {
                                          'units': env.action_space.n,
                                          'activation': 'softmax',
                                      }}}
            network_dict_advantage = None
            network_dict_value = None
    else:
        if dueling:
            network_dict_base = {0:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 32,
                                          'kernel_size': (8, 8),
                                          'strides': (4, 4),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 1:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 64,
                                          'kernel_size': (4, 4),
                                          'strides': (2, 2),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 2:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 64,
                                          'kernel_size': (3, 3),
                                          'strides': (1, 1),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }}}
            network_dict_advantage = {3:
                                          {'name': 'Flatten',
                                           'params': {}
                                           },
                                      4:
                                          {'name': 'Dense',
                                           'params': {
                                               'units': 512,
                                               'activation': 'relu',
                                               'kernel_initializer': tf.keras.initializers.HeNormal()
                                           }},
                                      5:
                                          {'name': 'Dense',
                                           'params': {
                                               'units': env.action_space.n,
                                               'activation': None
                                           }}}
            network_dict_value = {6:
                                      {'name': 'Flatten',
                                       'params': {}
                                       },
                                  7:
                                      {'name': 'Dense',
                                       'params': {
                                           'units': 512,
                                           'activation': 'relu',
                                           'kernel_initializer': tf.keras.initializers.HeNormal()
                                       }},
                                  8:
                                      {'name': 'Dense',
                                       'params': {
                                           'units': 1,
                                           'activation': None
                                       }}}
        else:
            network_dict_base = {0:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 32,
                                          'kernel_size': (8, 8),
                                          'strides': (4, 4),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 1:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 64,
                                          'kernel_size': (4, 4),
                                          'strides': (2, 2),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 2:
                                     {'name': 'Conv2D',
                                      'params': {
                                          'filters': 32,
                                          'kernel_size': (3, 3),
                                          'strides': (1, 1),
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 3:
                                     {'name': 'Flatten',
                                      'params': {}
                                      },
                                 4:
                                     {'name': 'Dense',
                                      'params': {
                                          'units': 512,
                                          'activation': 'relu',
                                          'kernel_initializer': tf.keras.initializers.HeNormal()
                                      }},
                                 5:
                                     {'name': 'Dense',
                                      'params': {
                                          'units': env.action_space.n,
                                          'activation': None,
                                      }}}
            network_dict_advantage = None
            network_dict_value = None
    return network_dict_base, network_dict_advantage, network_dict_value


def get_agent(env, network_dict_base, network_dict_advantage, network_dict_value, num_episodes, network_path):
    if network_dict_advantage is not None and network_dict_value is not None:
        dueling = True
        q_net_dict = [network_dict_base, network_dict_advantage, network_dict_value]
    else:
        dueling = False
        q_net_dict = network_dict_base

    return DDDQNAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      q_net_dict=q_net_dict,
                      q_target_net_dict=q_net_dict,
                      double_q=True,
                      dueling_q=dueling,
                      q_net_update=4,
                      q_target_net_update=10000,
                      discount_factor=0.99,
                      q_net_optimizer=tf.keras.optimizers.Adam,
                      q_target_net_optimizer=tf.keras.optimizers.Adam,
                      q_net_learning_rate=1e-4,
                      q_target_net_learning_rate=1e-4,
                      q_net_loss=tf.keras.losses.Huber(),
                      q_target_net_loss=tf.keras.losses.Huber(),
                      num_episodes=num_episodes,
                      learning_starts=1000,
                      memory_size=32768,
                      memory_alpha=0.7,
                      memory_beta=0.4,
                      max_epsilon=1.0,
                      min_epsilon=0.05,
                      epsilon_a=0.06,
                      epsilon_b=0.05,
                      epsilon_c=1.5,
                      batch_size=32,
                      max_norm_grad=10,
                      tau=1,
                      checkpoint_dir=network_path)


if __name__ == '__main__':
    for config in itertools.product(*[['ConnectFour'], ['Graphic'], ['MinMaxRandom'],
                                      ['Random']]):
        print('Executing the following config: {}'.format(config))
        algorithm = 'Dueling'
        environment = config[0]
        representation = config[1]
        opponent = config[2]
        agent_turn = config[3]
        num_episodes = 20000

        config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
        data_path = '..\\Results\\' + config_name + '\\'
        gif_path = data_path + 'GIFs\\'
        network_path = data_path + 'NetworkParameters\\'
        os.mkdir(data_path)
        os.mkdir(gif_path)
        os.mkdir(network_path)

        turn = get_turn(config)
        env = get_env(config, turn, representation, opponent)
        network_dict_base, network_dict_advantage, network_dict_value = get_network_dicts(representation, env, True)
        agent = get_agent(env, network_dict_base, network_dict_advantage,
                          network_dict_value, num_episodes, network_path)

        wizard = TurnGameTrainWizard(environment=environment,
                                     agent=agent,
                                     objective_score=1,
                                     running_average_length=100,
                                     evaluation_steps=100,
                                     evaluation_games=5,
                                     representation=representation,
                                     agent_turn=turn,
                                     agent_turn_test=None,
                                     opponent=opponent,
                                     path=data_path)

        wizard.train()