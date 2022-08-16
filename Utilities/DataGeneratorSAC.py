import itertools
import os
import tensorflow as tf

from Agents.SAC.SACAgent import SACAgent
from Utilities.TrainWizard import TurnGameTrainWizard
from Utilities.Wrappers import OpponentWrapper
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


def get_network_dicts(representation, env):
    if representation == 'Tabular':
        return None, None
    else:
        actor_dict_base = {0:
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
                                    'units': env.action_space.n,
                                    'activation': 'softmax'
                                }}}
        critic_dict_ext = {0:
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
                                    'units': env.action_space.n,
                                    'activation': None
                                }}}
    return actor_dict_base, critic_dict_ext


def get_agent(env, actor_dict_base, critic_dict_ext, num_episodes, network_path):
    return SACAgent(observation_space=env.observation_space,
                    action_space=env.action_space,
                    actor_net_dict=actor_dict_base,
                    critic_net_dict=critic_dict_ext,
                    net_update=1,
                    discount_factor=0.99,
                    actor_net_optimizer=tf.keras.optimizers.Adam,
                    critic_net_optimizer=tf.keras.optimizers.Adam,
                    actor_net_learning_rate=3e-4,
                    critic_net_learning_rate=3e-4,
                    actor_net_loss=tf.keras.losses.Huber(),
                    critic_net_loss=tf.keras.losses.Huber(),
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
                    batch_size=64,
                    max_norm_grad=5,
                    tau=0.005,
                    entropy_coeff=None,
                    initial_entropy_coeff=50.,
                    checkpoint_dir=network_path)


if __name__ == '__main__':
    for config in itertools.product(*[['ConnectFour'], ['Graphic'], ['Random'],
                                      ['Random']]):
        print('Executing the following config: {}'.format(config))
        algorithm = 'SAC'
        environment = config[0]
        representation = config[1]
        opponent = config[2]
        agent_turn = config[3]
        num_episodes = 10000

        config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
        data_path = '..\\Results\\' + config_name + '\\'
        gif_path = data_path + 'GIFs\\'
        network_path = data_path + 'NetworkParameters\\'
        os.mkdir(data_path)
        os.mkdir(gif_path)
        os.mkdir(network_path)

        turn = get_turn(config)
        env = get_env(config, turn, representation, opponent)
        actor_dict_base, critic_dict_ext = get_network_dicts(representation, env)
        agent = get_agent(env, actor_dict_base, critic_dict_ext, num_episodes, network_path)

        wizard = TurnGameTrainWizard(environment=environment,
                                     agent=agent,
                                     objective_score=1,
                                     running_average_length=100,
                                     evaluation_steps=50,
                                     evaluation_games=5,
                                     representation=representation,
                                     agent_turn=turn,
                                     agent_turn_test=None,
                                     opponent=opponent,
                                     path=data_path)

        wizard.train()
