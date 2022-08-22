import ast

from Utilities.MCTS import SelfPlayMCTS
from Agents.Agent import Agent
from Agents.SelfPlay.Network import SelfPlayNetwork
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.ConnectFour import ConnectFourEnv
from Utilities.MCTS import to_action
from multiprocessing import Pool
from os import cpu_count
import PIL.Image
import math
import random
import copy
import keras
import numpy as np

# CPU COUNT == 8
# INSERISCI MONTECARLO SIMULAZIONI A RUNTIME ( ANCHE EVALUATION )

class SelfPlayAgent(Agent):

    def __init__(self, observation_space, action_space, learning_rate, episodes=1000, iterations=1000, test_games=16, win_perc=0.55, mini_batches=100,
                 mcts_simulations = 640,
                 memory_size=262144, batch_size=32, checkpoint_dir='', multithreading = False):
        super(SelfPlayAgent, self).__init__(observation_space, action_space, batch_size, checkpoint_dir)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.iterations = iterations
        self.test_games = test_games
        self.win_perc = win_perc
        self.mcts_simulations = mcts_simulations
        self.mini_batches = mini_batches
        self.memory_size = memory_size
        self.multithreading = multithreading
        self.cpu_count = cpu_count()
        self.network = None

    def learn(self, env):
        nnet = SelfPlayNetwork(env, self.learning_rate).model
        memory = []
        for i in range(self.iterations):
            print("iteration: ", i)
            if self.multithreading:
                for j in range(math.ceil(self.episodes / self.cpu_count)):
                    print("episode: ", j * self.cpu_count, (j + 1) * self.cpu_count)
                    with Pool(self.cpu_count) as p:
                        memory.extend(p.map(self.play_episode, [(env, nnet, self.mcts_simulations) for _ in
                                                                range(self.cpu_count)]))
                        p.close()
            else:
                for j in range(self.episodes):
                    print("episode: ", j)
                    memory.extend(self.play_episode(env, nnet, self.mcts_simulations))

            nnet.save('tmp/nnet/' + env.name)  # saves compiled state
            new_nnet = keras.models.load_model('tmp/nnet/' + env.name)
            new_nnet = self.train_nnet(new_nnet, memory)  # Create copy of nnet with same weights as nnet

            if self.multithreading:
                for _ in range(math.ceil(self.test_games/self.cpu_count)):
                    with Pool(self.cpu_count) as p:
                        win_perc = p.map(self.compare_players, [(new_nnet, nnet, env)for _ in range(self.cpu_count)])
                        p.close()
                        win_perc = sum(win_perc)/len(win_perc)
            else:
                win_perc = 0
                for _ in range(self.test_games):
                    win_perc += self.compare_players(new_nnet, nnet, env)
                    win_perc = win_perc/self.test_games

            if win_perc > self.win_perc:
                nnet = new_nnet
        self.network = nnet
        return nnet

    def act(self, observation):
        self.network.predict(state_to_input_model(observation))

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
        while not done:
            mcts = get_mcts(env)
            for _ in range(self.mcts_simulations):
                mcts.rollout_simulation(env, first)
            action, mcts = mcts.select_next()
            obs, reward, done, info = env.step(ast.literal_eval(action))
            if done and challenger == first and reward == 1:  #challenger == first:  # conta il caso in cui la partita finisca per assenza di azioni
                challenger_wins += 1
                break
            elif done and reward == 0:
                challenger_wins +=1
                break
            elif done:
                break
            mcts = get_mcts(env)
            for _ in range(self.mcts_simulations):
                mcts.rollout_simulation(env, second)
            action, mcts = mcts.select_next()
            obs, reward, done, info = env.step(ast.literal_eval(action))
            if done and challenger == second and reward == -1: #challenger == second:
                challenger_wins += 1
                break
            if done and reward == 0:
                challenger_wins +=1
                break
        print("challenger won" if challenger_wins else "challenger lost")
        return challenger_wins

    #def sample_action(self, actions, env):
    #    action = np.argmax(actions)
    #    possible_actions = env.actions(env.state, env.player_one_workers, env.player_two_workers, env.player_one) if env.name == "Santorini" else env.actions(env.state)
    #    if len(possible_actions) == 0:
    #        return None
    #    while to_action(action, env.name) not in possible_actions: #while not env.check_valid_action(env.state, to_action(action, env.name), env.player_one_workers, env.player_two_workers, env.player_one):
    #        actions[action] = float("-inf")
    #        action = np.argmax(actions)
    #    return to_action(action, env.name)

    def play_episode(self, env, nnet, simulation_num=640):
        memories = []
        env.reset()
        mcts = get_mcts(env)
        turn = 0
        while True:
            print("turn N: ", turn)
            for _ in range(simulation_num):
                mcts.rollout_simulation(env, nnet)
            memories.append([mcts.get_relative_state(env.name, turn % 2), mcts.prior_actions, None])
            mcts = mcts.select_next()[1]
            env.render_board(mcts.state).show()
            mcts.parent = None  # drop the top part of the tree : Methods, Play
            turn += 1
            if env.goal(mcts.state)[1]:  # if game over
                memories = self.assign_reward(memories, (turn+1) % 2, env.goal(mcts.state)[0])  # turn+1%2 == winner
                return memories
            if env.name == "Santorini":
                if len(env.actions(mcts.state, mcts.player_one_workers, mcts.player_two_workers, mcts.player_one)) == 0: # Lost by no legal actions
                    memories = self.assign_reward(memories, not mcts.player_one)
                    return memories

    def assign_reward(self, memories, winner, reward = None):
        if reward is not None and reward == 0:
            for memory in memories:
                memory[2] = 0
        else:
            for memory in memories:
                memory[2] = 1 if winner else -1
                winner = not winner
        return memories

    def train_nnet(self, nnet, memory):
        for _ in range(self.mini_batches):
            minibatch = random.sample(memory, min(self.batch_size, len(memory)))

            training_states = {'input': np.array([row[0] for row in minibatch])}
            training_targets = {'policy_head': np.array([row[1] for row in minibatch])
                ,'value_head': np.array([row[2] for row in minibatch])
                }

            nnet.fit(training_states, training_targets, epochs=1, verbose=1, validation_split=0,
                                 batch_size=self.batch_size)
        return nnet


def state_to_input_model(state, name):
    if name == "TicTacToe":
        return state.reshape(-1, 3, 3, 1)
    elif name == "ConnectFour":
        return state.reshape(-1, 6, 7, 1)
    elif name == "Santorini":
        return state.reshape(-1, 5, 5, 6)


def get_mcts(env):
    if env.name == "Santorini":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=env.state,
                            player_one_workers=env.player_one_workers, player_two_workers=env.player_two_workers,
                            player_one=env.player_one)
    elif env.name == "ConnectFour":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=env.state)
    elif env.name == "TicTacToe":
        mcts = SelfPlayMCTS(prior_action=None, value=None, parent=None, state=env.state)
    return mcts

"""
if __name__ == '__main__':
    environment = TicTacToeEnv("Tabular", True)
    agent = SelfPlayAgent(observation_space=environment.observation_space, action_space=environment.action_space, batch_size=16,
                          checkpoint_dir='tmp/selfplayagent/' + environment.name, mini_batches=16, episodes=10, mcts_simulations=60, iterations=10, learning_rate=0.01)
    trained = agent.learn(environment)
    trained.save('tmp/trained/'+ environment.name)
    # Counter:
    # step -> time-step self-play
    # number update network
"""

if __name__ == '__main__':
    new_nnet = keras.models.load_model('tmp/trained/TicTacToe')
    env = TicTacToeEnv("Tabular", True)
    agent = SelfPlayAgent(observation_space=env.observation_space, action_space=env.action_space, batch_size=16,
                                checkpoint_dir='tmp/selfplayagent/' + env.name, mini_batches=16, episodes=30,
                                mcts_simulations=30, iterations=10, learning_rate=0.001)
    agent.model = new_nnet
    env.reset()
    env.render_board(env.state)
    #for _ in range(5):
    agent.play_episode(env, agent.model, 60)
