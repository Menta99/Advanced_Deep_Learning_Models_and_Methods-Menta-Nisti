import pickle

import numpy as np

from Utilities.ConnectFour import ConnectFourEnv
from Utilities.Santorini import SantoriniEnv
from Utilities.TicTacToe import TicTacToeEnv
from Wrappers import make_atari_test, wrap_deepmind, OpponentWrapper
from tqdm import tqdm
from PIL import Image


class TurnGameTrainWizard:
    def __init__(self, environment, agent, objective_score, running_average_length, evaluation_steps,
                 evaluation_games, representation, agent_turn, agent_turn_test, opponent, path):
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
        self.path = path
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
            return OpponentWrapper(SantoriniEnv(self.representation, agent_first), self.opponent)
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
        action = self.agent.act(state_init)
        state_next, reward, done, info, render, state_next_adv, reward_adv, done_adv, info_adv = self.environment.step(
            action[0])

        if done:
            self.episode_reward += reward
            self.agent.store(state_init, action, reward, state_next, done)
        elif done_adv:
            self.episode_reward += reward_adv
            self.agent.store(state_init, action, reward_adv, state_next, done)
        else:
            self.agent.store(state_init, action, reward, state_next, done)
            state_init = state_next_adv

        self.agent.learn()

        return state_init, done or done_adv

    def test_agent(self):
        print('Running average is {}'.format(np.mean(self.episode_reward_history)))
        f = open(self.path + 'scores.pkl', 'wb')
        results = self.play_test_games('full_game_{}'.format(self.index))
        self.eval_reward_history[self.evaluation_steps * self.index] = results
        print('Test Results:\nAverage Score: {}\nAverage Game Length: {}'
              .format(sum(i for i, _ in results)/self.evaluation_games, sum(j for _, j in results)/self.evaluation_games))
        print('Test Running Average:\nRunning Average Score: {}\nRunning Average Game Length: {}'
              .format(sum(i for t in list(self.eval_reward_history.values())[-self.running_average_length//self.evaluation_games:] for i, j in t)/
                      (min(len(self.eval_reward_history)*self.evaluation_games, self.running_average_length)),
                      sum(j for t in list(self.eval_reward_history.values())[-self.running_average_length//self.evaluation_games:] for i, j in t)/
                      (min(len(self.eval_reward_history)*self.evaluation_games, self.running_average_length))))
        pickle.dump(self.eval_reward_history, f)
        f.close()
        #self.agent.save() # keep an eye on memory, CNN are huge
        self.index += 1

    def update_stats(self):
        print(self.frame_count, self.episode_reward, self.agent._epsilon_scheduler(), self.environment.agent_first)
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
        for _ in tqdm(range(self.evaluation_games)):
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
            test_env = SantoriniEnv(self.environment.representation, agent_first)
        else:
            raise ValueError('Game provided does not exist!')
        test_env = OpponentWrapper(test_env, self.opponent)
        temp_time_step = self.agent.time_step
        temp_min_epsilon = self.agent.min_epsilon
        self.agent.time_step = self.agent.num_episodes
        self.agent.exploration_final_eps = 0
        init_state = test_env.reset()
        return test_env, temp_time_step, temp_min_epsilon, init_state

    def play_full_game(self, file_name, agent_first, gif):
        test_env, temp_time_step, temp_min_epsilon, state_init = self.init_test(agent_first)
        game_frame = []
        score = 0
        done = False
        done_adv = False
        while not done and not done_adv:
            game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
            action = self.agent.act(state_init)
            state_next, reward, done, info, render, state_next_adv, reward_adv, done_adv, info_adv = test_env.step(action[0])
            game_frame.append(render)
            if done:
                score += reward
            elif done_adv:
                game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
                score += reward_adv
            else:
                state_init = state_next_adv
        if gif:
            self.save_game_gif(game_frame, file_name, score)
        self.agent.time_step = temp_time_step
        self.agent.min_epsilon = temp_min_epsilon
        return score, len(game_frame)

    def save_game_gif(self, frames, file_name, score):
        print('Game len: ', len(frames), ' frames')
        print('Game score: ', score)
        #frames = [Image.fromarray(i) for i in frames]
        frames[0].save(self.path + 'GIFs\\' + file_name + '.gif', save_all=True, append_images=frames[1:], duration=500)
