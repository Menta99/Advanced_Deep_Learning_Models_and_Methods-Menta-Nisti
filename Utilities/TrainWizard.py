import pickle

import numpy as np

from Utilities.ConnectFour import ConnectFourEnv
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv
from termcolor import colored

from Utilities.Wrappers import OpponentWrapper


class TurnGameTrainWizard:
    def __init__(self, environment, agent, objective_score, running_average_length, evaluation_steps,
                 evaluation_games, representation, agent_turn, agent_turn_test, opponent, data_path,
                 gif_path, save_agent_checkpoints, montecarlo_init_sim=100000, montecarlo_normal_sim=25):
        self.agent = agent
        self.objective_score = objective_score
        self.running_average_length = running_average_length
        self.max_steps = self.agent.num_episodes
        self.evaluation_steps = evaluation_steps
        self.evaluation_games = evaluation_games
        self.representation = representation
        self.agent_turn = agent_turn
        self.agent_turn_test = agent_turn_test
        self.opponent = opponent
        self.data_path = data_path
        self.gif_path = gif_path
        self.save_agent_checkpoints = save_agent_checkpoints
        self.montecarlo_init_sim = montecarlo_init_sim
        self.montecarlo_normal_sim = montecarlo_normal_sim
        self.episode_reward = 0
        self.episode_reward_history = [-np.inf for _ in range(running_average_length)]
        self.games_played = 0
        self.eval_reward_history = {}
        self.rewards_sample = None
        self.frame_count = 0
        self.index = 1
        self.environment = self.build_env(environment)

    def build_env(self, environment_name):
        agent_first = self.agent_turn
        if self.agent_turn is None:
            agent_first = np.random.choice([True, False])
        if environment_name == 'TicTacToe':
            return OpponentWrapper(TicTacToeEnv(self.representation, agent_first), self.opponent)
        elif environment_name == 'ConnectFour':
            return OpponentWrapper(ConnectFourEnv(self.representation, agent_first), self.opponent)
        elif environment_name == 'Santorini':
            return OpponentWrapper(SantoriniEnv(self.representation, agent_first, True, True, self.montecarlo_init_sim,
                                                self.montecarlo_normal_sim), self.opponent)
        else:
            raise ValueError('Game provided does not exist!')

    def reset_env(self):
        if self.agent_turn is None:
            self.environment.agent_first = np.random.choice([True, False])
        state = self.environment.reset()
        self.episode_reward = 0
        self.rewards_sample = None
        self.games_played += 1
        return state

    def update_agent(self, state_init):
        action = self.agent.act(state_init, False)
        state_next, reward, done, info, render = self.environment.step(action[0])
        self.agent.store(state_init, action, reward, state_next, done)
        state_init = state_next
        if done:
            self.episode_reward += reward
            #print('Game finished: steps {} reward {}'.format(self.agent.time_step, reward))

        self.agent.learn()

        return state_init, done

    def test_agent(self):
        f = open(self.data_path + 'scores.pkl', 'wb')
        results = self.play_test_games('full_game_{}'.format(self.index))
        self.eval_reward_history[self.evaluation_steps * self.index] = results
        self.display_stats(results)
        pickle.dump(self.eval_reward_history, f)
        f.close()
        if self.save_agent_checkpoints:
            self.agent.save()
        self.index += 1

    def display_stats(self, results):
        text = 'History:\nTime Step: {} | Learning Step: {}\nEpsilon: {} | Last Loss: {}\n' \
               'Test Results:\nAverage Score: {:.2f}\nAverage Game Length: {:.2f}\nWins: ' \
               '{} | Losses: {} | Ties: {} | Invalid: {}' \
               '\nTest Running Average:\nRunning Average Score: {:.2f}\nRunning Average Game Length: {:.2f}'. \
            format(self.agent.time_step, self.agent.learn_step_counter,
                   self.agent.epsilon_scheduler(), self.agent.last_loss,
                   sum(i for i, _ in results) / self.evaluation_games,
                   sum(j for _, j in results) / self.evaluation_games,
                   sum(1 for i, _ in results if i == 1), sum(1 for i, _ in results if i == -1),
                   sum(1 for i, _ in results if i == 0), sum(1 for i, _ in results if i == -2),
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

    def update_stats(self):
        self.episode_reward_history[(self.games_played - 1) % self.running_average_length] = self.episode_reward
        if np.mean(self.episode_reward_history) > self.objective_score:
            print("Solved at episode {}!".format(self.agent.time_step))
            return True
        return False

    def train(self):
        while self.frame_count <= self.max_steps:
            state_init = self.reset_env()
            while True:
                self.frame_count += 1

                state_init, game_over = self.update_agent(state_init)
                if game_over:
                    break

                if self.frame_count > self.evaluation_steps * self.index:
                    self.test_agent()

            if self.update_stats():
                break

    def play_test_games(self, file_name):
        scores = []
        agent_first = self.agent_turn_test
        for _ in range(self.evaluation_games):
            if self.agent_turn_test is None:
                agent_first = np.random.choice([True, False])
            scores.append(self.play_full_game(None, agent_first, False))
        self.play_full_game(file_name, agent_first, True)
        return scores

    def init_test(self, agent_first):
        if self.environment.name == 'TicTacToe':
            test_env = TicTacToeEnv(self.environment.representation, agent_first)
        elif self.environment.name == 'ConnectFour':
            test_env = ConnectFourEnv(self.environment.representation, agent_first)
        elif self.environment.name == 'Santorini':
            test_env = SantoriniEnv(self.environment.representation, agent_first, True, False, 0,
                                    self.montecarlo_normal_sim)
        else:
            raise ValueError('Game provided does not exist!')
        test_env = OpponentWrapper(test_env, self.opponent)
        if self.environment.name == "Santorini":
            root = self.environment.mc_node
            while not root.is_root():
                root = root.parent
            test_env.mc_node = root
            test_env.mcts = True
        init_state = test_env.reset()
        return test_env, init_state

    def play_full_game(self, file_name, agent_first, gif):
        test_env, state_init = self.init_test(agent_first)
        game_frame = []
        score = 0
        done = False
        while not done:
            game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
            action = self.agent.act(state_init, True)
            state_next, reward, done, info, render = test_env.step(action[0])
            state_init = state_next
            game_frame.append(render)
            if done:
                score += reward
        game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
        if gif:
            self.save_game_gif(game_frame, file_name)
        return score, len(game_frame)

    def save_game_gif(self, frames, file_name):
        frames[0].save(self.gif_path + file_name + '.gif', save_all=True, append_images=frames[1:], duration=500)
