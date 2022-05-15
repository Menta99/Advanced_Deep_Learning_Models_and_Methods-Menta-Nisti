import numpy as np
from Wrappers import make_atari_test, wrap_deepmind
from PIL import Image


class TrainWizard:
    def __init__(self, environment, agent, objective_score, running_average_length,
                 evaluation_steps, path='GIFs/'):
        self.environment = environment
        self.agent = agent
        self.objective_score = objective_score
        self.running_average_length = running_average_length
        self.max_steps = self.agent.num_episodes
        self.evaluation_steps = evaluation_steps
        self.path = path
        self.episode_reward = 0
        self.episode_reward_history = []
        self.rewards_sample = None
        self.frame_count = 0

    def train(self):
        while True:
            state_init = self.environment.reset()
            self.episode_reward = 0
            self.rewards_sample = None
            while True:
                self.frame_count += 1
                action = self.agent.act(state_init)
                state_next, reward, done, _ = self.environment.step(action[0])
                self.episode_reward += reward

                self.agent.store(state_init, action, reward, state_next, done)
                state_init = state_next
                self.agent.learn()

                if done:
                    break

                if self.frame_count % self.evaluation_steps == 0:
                    self.play_full_game('full_game_' + str(self.frame_count // 1000) + 'K')

            if reward is not None:
                print(self.frame_count, self.episode_reward, self.agent._epsilon_scheduler())

            self.episode_reward_history.append(self.episode_reward)
            if len(self.episode_reward_history) > self.running_average_length:
                del self.episode_reward_history[:1]

            if np.mean(self.episode_reward_history) > self.objective_score:
                print("Solved at episode {}!".format(self.agent.time_step))
                break

    def play_full_game(self, file_name):
        test_env = make_atari_test(self.environment.spec.id)
        test_env = wrap_deepmind(test_env, frame_stack=True, scale=True)
        test_env.seed(np.random.randint(0, 10000))
        temp_time_step = self.agent.time_step
        temp_min_epsilon = self.agent.min_epsilon
        self.agent.time_step = self.agent.num_episodes
        self.agent.exploration_final_eps = 0
        game_frame = []
        score = 0
        for _ in range(test_env.unwrapped.ale.lives()):
            init_state = test_env.reset()
            terminal = False
            while not terminal:
                game_frame.append(test_env.render(mode='rgb_array'))
                sel_action = self.agent.act(init_state)
                next_state, rw, terminal, info = test_env.step(sel_action[0])
                score += rw
                init_state = next_state
        print('Game len: ', len(game_frame), ' frames')
        print('Game score: ', score)
        self.save_game_gif(game_frame, file_name)
        self.agent.time_step = temp_time_step
        self.agent.min_epsilon = temp_min_epsilon

    def save_game_gif(self, frames, file_name):
        images = [Image.fromarray(j) for j in frames]
        images[0].save(self.path + file_name + '.gif', save_all=True, append_images=images[1:])
