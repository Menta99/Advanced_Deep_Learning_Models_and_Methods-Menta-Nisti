import ast
import os
import pickle

from termcolor import colored

from Utilities.MCTS import SelfPlayMCTS
from Agents.Agent import Agent
from Agents.SelfPlay.Network import SelfPlayNetwork
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.ConnectFour import ConnectFourEnv
from Utilities.MCTS import to_action, state_to_input_model
from os import cpu_count
import gc
import sys
import PIL.Image
import math
import random
import copy
import keras
import numpy as np
import time
from tqdm import tqdm
from Utilities.Wrappers import OpponentWrapper


class SelfPlayAgent(Agent):

    def __init__(self, observation_space, action_space, learning_rate, episodes=1000, iterations=1000, test_games=16,
                 win_perc=0.55, tau=0,
                 mini_batches=100, evaluation_steps=100, evaluation_games=10,
                 running_average_length=100, gif_path='', opponent="Random",
                 data_path='', memory_size = 2048,
                 mcts_simulations=640, agent_turn_test=None,
                 batch_size=32, checkpoint_dir='', multithreading=False,
                 tmp_path = "tmp/nnet"):
        super(SelfPlayAgent, self).__init__(observation_space, action_space, batch_size, checkpoint_dir)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.iterations = iterations
        self.test_games = test_games
        self.win_perc = win_perc
        self.tau = tau
        self.mcts_simulations = mcts_simulations
        self.mini_batches = mini_batches
        self.evaluation_steps = evaluation_steps
        self.evaluation_games = evaluation_games
        self.multithreading = multithreading
        self.memory = []
        self.memory_size = memory_size
        self.index = 1
        self.num_updates = 0
        self.cpu_count = cpu_count()
        self.network = None
        self.eval_reward_history = {}
        self.time_step = 0
        self.running_average_length = running_average_length
        self.agent_turn_test = agent_turn_test
        self.gif_path = gif_path
        self.data_path = data_path
        self.tmp_path = tmp_path
        self.opponent = opponent

    def learn(self, env):
        self.network = SelfPlayNetwork(env, self.learning_rate).model
        for i in range(self.iterations):
            print("iteration: ", i)
            if self.multithreading:
                for j in range(math.ceil(self.episodes / self.cpu_count)):
                    with Pool(self.cpu_count) as p:
                        self.memory.extend(p.map(self.play_episode, [(env, self.network, self.mcts_simulations) for _ in
                                                                     range(self.cpu_count)]))
                        p.close()
            else:
                for _ in tqdm(range(self.episodes)):
                    self.memory.extend(self.play_episode(env, self.network, self.mcts_simulations, self.tau))
                    while len(self.memory) > self.memory_size:
                        self.memory.pop(0)

            self.network.save(self.tmp_path)  # saves compiled state
            new_nnet = keras.models.load_model(self.tmp_path)
            new_nnet = self.train_nnet(new_nnet, self.memory)  # Create copy of nnet with same weights as nnet

            if self.multithreading:
                for _ in range(math.ceil(self.test_games / self.cpu_count)):
                    with Pool(self.cpu_count) as p:
                        win_perc = p.map(self.compare_players,
                                         [(new_nnet, self.network, env) for _ in range(self.cpu_count)])
                        p.close()
                        win_perc = sum(win_perc) / len(win_perc)
            else:
                win_perc = 0
                print("Comparing Networks ...", flush=True)
                for _ in tqdm(range(self.test_games)):
                    win_perc += self.compare_players(new_nnet, self.network, env)
                win_perc = ((win_perc / self.test_games) + 1)/2

            if win_perc > self.win_perc:
                print("Updated Network", flush=True)
                self.network = new_nnet
                self.num_updates += 1
            if self.time_step > self.evaluation_steps * self.index:
                self.test_agent(env)
        return self.network

    def act(self, observation, env, agent_first, mcts):
        #mcts = get_mcts(env)
        #mcts.state = mcts.get_relative_state(env.state, env)
        for _ in range(self.mcts_simulations):
            mcts.rollout_simulation(env, self.network)
        actions = mcts.get_probs(env.action_space.n, env)
        action = np.argmax(actions)
        # action = np.where(actions == np.random.choice(actions, p=actions))
        action = to_action(action, env.name)
        # action, _ = mcts.select_next()
        return action  # ast.literal_eval(action)

    def store(self, initial_state, action, reward, final_state, terminal):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def compare_players(self, challenger, defender, env):
        challenger_wins = 0
        env.reset()
        done = False
        first, second = (challenger, defender) if random.randint(0, 1) == 0 else (defender, challenger)
        mcts_first = get_mcts(env)
        mcts_second = get_mcts(env)
        action_one = None
        action_two = None
        while not done:
            if mcts_first.is_leaf():
                mcts_first.expand(env)
                mcts_first.evaluate(first, env)
            if action_two is not None:
                mcts_first = mcts_first.children[str(action_two)]
            for _ in range(self.mcts_simulations):
                mcts_first.rollout_simulation(env, first)
            actions = mcts_first.get_probs(env.action_space.n, env)
            action_one = np.argmax(actions)
            action_one = to_action(action_one, env.name)
            obs, reward, done, info = env.step(action_one)
            mcts_first = mcts_first.children[str(action_one)]
            if done:
                if reward == 1 and challenger == first:
                    challenger_wins = 1
                elif reward == -1 and challenger == first:
                    challenger_wins = -1
                elif reward == 1 and challenger == second:
                    challenger_wins = -1
                elif reward == -1 and challenger == second:
                    challenger_wins = 1
                elif reward == -2:
                    print("INVALID ACTION WTF")
                break
            if mcts_second.is_leaf():
                mcts_second.expand(env)
                mcts_second.evaluate(second, env)
            if action_one is not None:
                mcts_second = mcts_second.children[str(action_one)]
            for _ in range(self.mcts_simulations):
                mcts_second.rollout_simulation(env, second)
            actions = mcts_second.get_probs(env.action_space.n, env)
            action_two = np.argmax(actions)  # np.where(actions == np.random.choice(actions, p=actions))
            action_two = to_action(action_two, env.name)
            obs, reward, done, info = env.step(action_two)
            mcts_second = mcts_second.children[str(action_two)]
            if done:
                if reward == 1 and challenger == first:
                    challenger_wins = 1
                elif reward == -1 and challenger == first:
                    challenger_wins = -1
                elif reward == 1 and challenger == second:
                    challenger_wins = -1
                elif reward == -1 and challenger == second:
                    challenger_wins = 1
                elif reward == -2:
                    print("INVALID ACTION WTF")
                break
        print(challenger_wins)
        return challenger_wins

    def play_episode(self, env, nnet, simulation_num=640, tau=0):
        memories = []
        env.reset()
        mcts = get_mcts(env)
        turn = 0
        while True:
            # print("turn N: ", turn)
            for _ in range(simulation_num):
                mcts.rollout_simulation(env, nnet)
            memories.append(
                [mcts.state, mcts.get_probs(env.action_space.n, env), None])
            actions = mcts.get_probs(env.action_space.n, env)
            if tau > 0:
                action = np.where(actions == np.random.choice(actions, p=actions))[0][0]
                tau = tau - 1
            else:
                action = np.argmax(actions)
            action = to_action(action, env.name)
            mcts = mcts.children[str(action)]
            self.time_step += 1
            mcts.parent.children = None
            mcts.parent = None
            turn += 1
            if env.name == "TicTacToe" or env.name == "ConnectFour":
                if env.goal(mcts.state)[1]:  # if game over
                    memories = self.assign_reward(memories, env.goal(mcts.state)[0])  # turn+1%2 == winner
                    break
            if env.name == "Santorini":
                if env.goal(mcts.state, mcts.player_one_workers, mcts.player_two_workers, mcts.player_one)[1]:
                #if len(env.actions(mcts.state, mcts.player_one_workers, mcts.player_two_workers,
                #                   mcts.player_one)) == 0:  # Lost by no legal actions
                    memories = self.assign_reward(memories, env.goal(mcts.state, mcts.player_one_workers, mcts.player_two_workers, mcts.player_one)[0])
                    break
        return memories

    def assign_reward(self, memories, reward=None):
        memories.reverse()
        if reward is not None and reward == 0:
            for memory in memories:
                memory[2] = 0
        else:
            reward = -reward # Reward -1
            for memory in memories:
                memory[2] = reward  # 1 if winner == 0 else -1
                reward = -reward
                # winner = not winner
        return memories

    def train_nnet(self, nnet, memory):  # Convertire memory[0] to Image se Graphic
        print("Training ...", flush=True)
        for _ in tqdm(range(self.mini_batches)):
            minibatch = random.sample(memory, min(self.batch_size, len(memory)))
            training_states = {'input': np.array([row[0] for row in minibatch])}
            training_targets = {'policy_head': np.array([row[1] for row in minibatch])
                , 'value_head': np.array([row[2] for row in minibatch])
                                }

            nnet.fit(training_states, training_targets, epochs=1, verbose=0, validation_split=0,
                     batch_size=self.batch_size)
        return nnet

    def test_agent(self, env):
        f = open(self.data_path + 'scores.pkl', 'wb')
        results = self.play_test_games('full_game_{}'.format(self.index), env)
        self.eval_reward_history[self.evaluation_steps * self.index] = results
        self.display_stats(results)
        pickle.dump(self.eval_reward_history, f)
        f.close()
        self.index += 1

    def init_test(self, agent_first, env):
        if env.name == 'TicTacToe':
            test_env = TicTacToeEnv(env.representation, agent_first)
        elif env.name == 'ConnectFour':
            test_env = ConnectFourEnv(env.representation, agent_first)
        elif env.name == 'Santorini':
            test_env = SantoriniEnv(env.representation, agent_first)
        else:
            raise ValueError('Game provided does not exist!')
        test_env = OpponentWrapper(test_env, self.opponent)
        if not agent_first:
            init_state, action = test_env.reset(True)
            return test_env, init_state, action
        init_state = test_env.reset()
        return test_env, init_state

    def play_test_games(self, file_name, env):
        scores = []
        agent_first = self.agent_turn_test
        print("Playing Test Games vs", self.opponent, flush=True)
        for _ in tqdm(range(self.evaluation_games)):
            if self.agent_turn_test is None:
                agent_first = np.random.choice([True, False])
            scores.append(self.play_full_game(None, agent_first, False, env))
        self.play_full_game(file_name, agent_first, True, env)
        return scores

    def play_full_game(self, file_name, agent_first, gif, env):
        if not agent_first:
            test_env, state_init, action = self.init_test(agent_first, env)
        else:
            test_env, state_init = self.init_test(agent_first, env)
        mcts = get_mcts(test_env)
        #if not agent_first:
        #    mcts.expand(test_env)
        #    mcts.evaluate(self.network, test_env)
        #    mcts = mcts.children[str(action)]
        game_frame = []
        score = 0
        done = False
        while not done:
            game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
            action = self.act(state_init, test_env, agent_first, mcts)
            mcts = mcts.children[str(action)]
            state_next, reward, done, info, render, opponent_action = test_env.step(action, True)
            if opponent_action is not None:
                if mcts.is_leaf():
                    mcts.expand(test_env)
                    mcts.evaluate(self.network, test_env)
                mcts = mcts.children[str(opponent_action)]
            state_init = state_next
            game_frame.append(render)
            if done:
                score += reward
        game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
        print(agent_first, reward)
        if gif:
            self.save_game_gif(game_frame, file_name)
        return score, len(game_frame)

    def display_stats(self, results):
        text = 'History:\nTime Step: {} | Num Updates: {}\n' \
               'Test Results:\nAverage Score: {:.2f}\nAverage Game Length: {:.2f}\nWins: ' \
               '{} | Losses: {} | Ties: {}' \
               '\nTest Running Average:\nRunning Average Score: {:.2f}\nRunning Average Game Length: {:.2f}'. \
            format(self.time_step,
                   self.num_updates,

                   sum(i for i, _ in results) / self.evaluation_games,
                   sum(j for _, j in results) / self.evaluation_games,
                   sum(1 for i, _ in results if i == 1), sum(1 for i, _ in results if i == -1),
                   sum(1 for i, _ in results if i == 0),
                   sum(i for t in
                       list(self.eval_reward_history.values())[-self.running_average_length // self.evaluation_games:]
                       for i, j in t) / (
                       min(len(self.eval_reward_history) * self.evaluation_games, self.running_average_length)),
                   sum(j for t in
                       list(self.eval_reward_history.values())[-self.running_average_length // self.evaluation_games:]
                       for i, j in t) / (
                       min(len(self.eval_reward_history) * self.evaluation_games, self.running_average_length)))
        lines = text.splitlines()
        width = max(len(s) for s in lines)
        res = ['┌' + '─' * (width + 2) + '┐']
        for s in lines:
            res.append('│ ' + (s + ' ' * width)[:width] + ' │')
        res.append('└' + '─' * (width + 2) + '┘')
        print(colored('\n'.join(res), 'green'))

    def save_game_gif(self, frames, file_name):
        frames[0].save(self.gif_path + file_name + '.gif', save_all=True, append_images=frames[1:], duration=500)


def get_mcts(env):
    if env.name == "Santorini":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=copy.copy(env.state),
                            player_one_workers=copy.copy(env.player_one_workers), player_two_workers=copy.copy(env.player_two_workers),
                            player_one=env.player_one)
    elif env.name == "ConnectFour":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=copy.copy(env.state))
    elif env.name == "TicTacToe":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=copy.copy(env.state))
    return mcts


if __name__ == '__main__':

    # config_name = algorithm + '_' + environment + '_' + representation + '_' + opponent + '_' + agent_turn
    config_name = "SelfPlay\\TicTacToe"
    data_path = '..\\..\\FinalResults\\' + config_name + '\\'
    gif_path = data_path + 'GIFs\\'
    network_path = data_path + 'NetworkParameters\\'  # Final Network
    tmp_path = data_path + "tmp\\"
    os.mkdir(data_path)
    os.mkdir(tmp_path)
    os.mkdir(gif_path)
    os.mkdir(network_path)

    environment = TicTacToeEnv("Tabular", True)
    agent = SelfPlayAgent(observation_space=environment.observation_space, action_space=environment.action_space,
                          learning_rate=0.01, episodes=64, iterations=50, test_games=11,
                          win_perc=0.55, tau=4, memory_size=10000,
                          mini_batches=96, evaluation_steps=1, evaluation_games=10,
                          running_average_length=100, gif_path=gif_path, opponent="MinMaxRandom",
                          data_path=data_path,
                          mcts_simulations=50, agent_turn_test=None,
                          batch_size=32, checkpoint_dir=network_path, multithreading=False,
                          tmp_path=tmp_path)
    trained = agent.learn(environment)
    trained.save(network_path)

"""
    config_name = "SelfPlay\\ConnectFour"
    data_path = '..\\..\\FinalResults\\' + config_name + '\\'
    gif_path = data_path + 'GIFs\\'
    network_path = data_path + 'NetworkParameters\\'  # Final Network
    os.mkdir(data_path)
    os.mkdir(gif_path)
    os.mkdir(network_path)

    environment = ConnectFourEnv("Tabular", True)
    agent = SelfPlayAgent(observation_space=environment.observation_space, action_space=environment.action_space,
                          learning_rate=0.01, episodes=16, iterations=60, test_games=11,
                          win_perc=0.55, tau=7, memory_size=18000,
                          mini_batches=256, evaluation_steps=1, evaluation_games=10,
                          running_average_length=100, gif_path=gif_path, opponent="MinMaxRandom",
                          data_path=data_path,
                          mcts_simulations=60, agent_turn_test=None,
                          batch_size=32, checkpoint_dir=network_path, multithreading=False)
    trained = agent.learn(environment)
    trained.save('tmp/trained/' + environment.name)
"""

"""
if __name__ == '__main__':
    # new_nnet = keras.models.load_model('tmp/nnet/TicTacToe')
    env = TicTacToeEnv("Tabular", True)
    agent = SelfPlayAgent(observation_space=env.observation_space, action_space=env.action_space, batch_size=16,
                          checkpoint_dir='tmp/selfplayagent/' + env.name, mini_batches=16, episodes=30,
                          mcts_simulations=10, iterations=10, learning_rate=0.001)
    agent.model = SelfPlayNetwork(env, 0.01)
    env.reset()
    # env.render_board(env.state)
    # for _ in range(5):
    memories = agent.play_episode(env, agent.model, 10)
    for memory in memories:
        env.render_board(memory[0]).show()
        print(memory[2])
"""
