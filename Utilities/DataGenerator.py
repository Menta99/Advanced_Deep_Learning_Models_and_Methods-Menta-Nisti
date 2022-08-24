import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import os
import sys

from Agents.DDDQN.DDDQNAgent import DDDQNAgent
from Agents.SAC.SACAgent import SACAgent
from Utilities.ConnectFour import ConnectFourEnv
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.TrainWizard import TurnGameTrainWizard
from Utilities.Wrappers import OpponentWrapper


def get_turn(config_turn):
    if config_turn == 'First':
        return True
    elif config_turn == 'Second':
        return False
    elif config_turn == 'Random':
        return None
    else:
        raise ValueError('Turn provided does not exist!')


def get_env(config_environment, config_representation, config_turn, config_opponent):
    if config_environment == 'TicTacToe':
        return OpponentWrapper(TicTacToeEnv(config_representation, config_turn), config_opponent)
    elif config_environment == 'ConnectFour':
        return OpponentWrapper(ConnectFourEnv(config_representation, config_turn), config_opponent)
    elif config_environment == 'Santorini':
        return OpponentWrapper(SantoriniEnv(config_representation, config_turn, True, False, 0, 0), config_opponent)
    else:
        raise ValueError('Game provided does not exist!')


def get_network_dicts(config_algorithm, config_representation, config_environment):
    if config_algorithm == 'DDDQN':
        if config_representation == 'Tabular':
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
                                     {'name': 'Dense',
                                      'params': {
                                          'units': 16,
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
                                               'units': config_environment.action_space.n,
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
            return [network_dict_base, network_dict_advantage, network_dict_value]
        elif config_representation == 'Graphic':
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
                                               'units': config_environment.action_space.n,
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
            return [network_dict_base, network_dict_advantage, network_dict_value]
        else:
            raise ValueError('Representation provided does not exist!')
    elif config_algorithm == 'SAC':
        if config_representation == 'Tabular':
            actor_dict = {0:
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
                              {'name': 'Dense',
                               'params': {
                                   'units': 16,
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
                                   'units': config_environment.action_space.n,
                                   'activation': 'softmax'
                               }}}
            critic_dict = {0:
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
                               {'name': 'Dense',
                                'params': {
                                    'units': 16,
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
                                    'units': config_environment.action_space.n,
                                    'activation': None
                                }}}
            return [actor_dict, critic_dict]
        elif config_representation == 'Graphic':
            actor_dict = {0:
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
                                   'units': config_environment.action_space.n,
                                   'activation': 'softmax'
                               }}}
            critic_dict = {0:
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
                                    'units': config_environment.action_space.n,
                                    'activation': None
                                }}}
            return [actor_dict, critic_dict]
        else:
            raise ValueError('Representation provided does not exist!')
    else:
        raise ValueError('Algorithm provided does not exist!')


def get_agent(config_env, config_algorithm, config_network_dicts, config_network_path, config_test_params):
    if config_algorithm == 'DDDQN':
        return DDDQNAgent(observation_space=config_env.observation_space,
                          action_space=config_env.action_space,
                          q_net_dict=config_network_dicts,
                          q_target_net_dict=config_network_dicts,
                          double_q=True,
                          dueling_q=True,
                          q_net_update=4,
                          q_target_net_update=10000,
                          discount_factor=0.99,
                          q_net_optimizer=tf.keras.optimizers.Adam,
                          q_target_net_optimizer=tf.keras.optimizers.Adam,
                          q_net_learning_rate=1e-4,
                          q_target_net_learning_rate=1e-4,
                          q_net_loss=tf.keras.losses.Huber(),
                          q_target_net_loss=tf.keras.losses.Huber(),
                          num_episodes=config_test_params['num_episodes'],
                          learning_starts=config_test_params['learning_starts'],
                          memory_size=config_test_params['memory_size'],
                          memory_alpha=config_test_params['memory_alpha'],
                          memory_beta=config_test_params['memory_beta'],
                          max_epsilon=config_test_params['max_epsilon'],
                          min_epsilon=config_test_params['min_epsilon'],
                          epsilon_a=config_test_params['epsilon_a'],
                          epsilon_b=config_test_params['epsilon_b'],
                          epsilon_c=config_test_params['epsilon_c'],
                          batch_size=config_test_params['batch_size'],
                          max_norm_grad=10,
                          tau=1,
                          checkpoint_dir=config_network_path)
    elif config_algorithm == 'SAC':
        return SACAgent(observation_space=config_env.observation_space,
                        action_space=config_env.action_space,
                        actor_net_dict=config_network_dicts[0],
                        critic_net_dict=config_network_dicts[1],
                        net_update=1,
                        discount_factor=0.99,
                        actor_net_optimizer=tf.keras.optimizers.Adam,
                        critic_net_optimizer=tf.keras.optimizers.Adam,
                        actor_net_learning_rate=3e-4,
                        critic_net_learning_rate=3e-4,
                        actor_net_loss=tf.keras.losses.Huber(),
                        critic_net_loss=tf.keras.losses.Huber(),
                        num_episodes=config_test_params['num_episodes'],
                        learning_starts=config_test_params['learning_starts'],
                        memory_size=config_test_params['memory_size'],
                        memory_alpha=config_test_params['memory_alpha'],
                        memory_beta=config_test_params['memory_beta'],
                        max_epsilon=config_test_params['max_epsilon'],
                        min_epsilon=config_test_params['min_epsilon'],
                        epsilon_a=config_test_params['epsilon_a'],
                        epsilon_b=config_test_params['epsilon_b'],
                        epsilon_c=config_test_params['epsilon_c'],
                        batch_size=config_test_params['batch_size'],
                        max_norm_grad=5,
                        tau=0.005,
                        entropy_coeff=None,
                        initial_entropy_coeff=50.,
                        checkpoint_dir=config_network_path)
    else:
        raise ValueError('Algorithm provided does not exist!')


if __name__ == '__main__':
    for config in itertools.product(*[['DDDQN'], ['Santorini'],
                                      ['Tabular', 'Graphic'], ['MonteCarlo']]):
        print('Executing the following config: {}'.format(config))
        algorithm = config[0]
        environment = config[1]
        representation = config[2]
        opponent = config[3]
        agent_turn = 'Random'
        test_params = {
            'num_episodes': 200000,
            'learning_starts': 1000,
            'memory_size': 32768,
            'memory_alpha': 0.7,
            'memory_beta': 0.4,
            'max_epsilon': 1.0,
            'min_epsilon': 0.05,
            'epsilon_a': 0.06,
            'epsilon_b': 0.05,
            'epsilon_c': 1.5,
            'batch_size': 32
        }

        if (environment in ['TicTacToe', 'ConnectFour'] and opponent in ['MonteCarlo']) or (
                environment in ['Santorini'] and opponent in ['MinMaxRandom']):
            print('Config {} not supported, skipping...'.format(config))
            continue

        config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
        data_path = '..\\RedoResults\\' + config_name + '\\'
        gif_path = data_path + 'GIFs\\'
        network_path = data_path + 'NetworkParameters\\'
        os.mkdir(data_path)
        os.mkdir(gif_path)
        os.mkdir(network_path)

        turn = get_turn(agent_turn)
        env = get_env(environment, representation, turn, opponent)
        network_dicts = get_network_dicts(algorithm, representation, env)
        agent = get_agent(env, algorithm, network_dicts, network_path, test_params)

        wizard = TurnGameTrainWizard(environment=environment,
                                     agent=agent,
                                     objective_score=1,
                                     running_average_length=100,
                                     evaluation_steps=200,
                                     evaluation_games=5,
                                     representation=representation,
                                     agent_turn=turn,
                                     agent_turn_test=None,
                                     opponent=opponent,
                                     data_path=data_path,
                                     gif_path=gif_path,
                                     save_agent_checkpoints=False,
                                     montecarlo_init_sim=100000,
                                     montecarlo_normal_sim=10)

        wizard.train()
        wizard.agent.save()
