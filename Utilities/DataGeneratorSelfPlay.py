import os

from Agents.SelfPlay.Agent import SelfPlayAgent
from Utilities.ConnectFour import ConnectFourEnv
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv


def get_env(config_environment, config_representation):
    setup = {
        "learning_rate": 0.01,
        "test_games": 10,
        "win_perc": 0.55,
        "evaluation_steps": 1,
        "running_average_length": 100,
        "gif_path": gif_path,
        "data_path": data_path,
        "agent_turn_test": None,
        "checkpoint_dir": network_path,
        "multithreading": False,
        "tmp_path": tmp_path
    }
    if config_environment == 'TicTacToe':
        setup['episodes'] = 64
        setup['iterations'] = 25
        setup['tau'] = 0
        setup['memory_size'] = 2560
        setup['mini_batches'] = 32
        setup['evaluation_games'] = 10
        setup['opponent'] = 'MinMaxRandom'
        setup['mcts_simulations'] = 100
        setup['batch_size'] = 16
        return TicTacToeEnv(config_representation, True), setup
    elif config_environment == 'ConnectFour':
        setup['episodes'] = 20
        setup['iterations'] = 25
        setup['tau'] = 10
        setup['memory_size'] = 2500
        setup['mini_batches'] = 28
        setup['evaluation_games'] = 10
        setup['opponent'] = 'MinMaxRandom'
        setup['mcts_simulations'] = 100
        setup['batch_size'] = 16
        return ConnectFourEnv(config_representation, True), setup
    elif config_environment == 'Santorini':
        setup['episodes'] = 20
        setup['iterations'] = 25
        setup['tau'] = 12
        setup['memory_size'] = 3000
        setup['mini_batches'] = 16
        setup['evaluation_games'] = 5
        setup['test_games'] = 5
        setup['opponent'] = 'MonteCarlo'
        setup['mcts_simulations'] = 400
        setup['batch_size'] = 32
        return SantoriniEnv(config_representation, True), setup


if __name__ == '__main__':
    representation = "Tabular"
    env_name = "ConnectFour"
    config_name = 'SelfPlay-' + env_name
    data_path = '..\\Results\\' + config_name + '\\'
    gif_path = data_path + 'GIFs\\'
    tmp_path = data_path + "tmp\\"
    network_path = data_path + 'NetworkParameters\\'
    os.mkdir(data_path)
    os.mkdir(gif_path)
    os.mkdir(tmp_path)
    os.mkdir(network_path)

    environment, train_setup = get_env(env_name, representation)
    agent = SelfPlayAgent(observation_space=environment.observation_space, action_space=environment.action_space,
                          learning_rate=train_setup["learning_rate"],
                          episodes=train_setup["episodes"],
                          iterations=train_setup["iterations"],
                          test_games=train_setup["test_games"],
                          win_perc=train_setup["win_perc"],
                          tau=train_setup["tau"],
                          memory_size=train_setup["memory_size"],
                          mini_batches=train_setup["mini_batches"],
                          evaluation_steps=train_setup["evaluation_steps"],
                          evaluation_games=train_setup["evaluation_games"],
                          running_average_length=train_setup["running_average_length"],
                          gif_path=train_setup["gif_path"],
                          opponent=train_setup["opponent"],
                          data_path=train_setup["data_path"],
                          mcts_simulations=train_setup["mcts_simulations"],
                          agent_turn_test=train_setup["agent_turn_test"],
                          batch_size=train_setup["batch_size"],
                          checkpoint_dir=train_setup["checkpoint_dir"],
                          multithreading=train_setup["multithreading"],
                          tmp_path=train_setup["tmp_path"]
                          )
    trained_net = agent.learn(environment)
    trained_net.save(network_path)
