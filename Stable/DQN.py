import sys
import time
from collections import deque

from gym import Wrapper
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from termcolor import colored

from Stable.Buffer import ReplayBuffer
from Utilities.TicTacToe import TicTacToeEnv
from Utilities.Wrappers import OpponentWrapper
from Logger import Logger, configure_logger
from PIL import Image


class DQN:
    def __init__(self, env, learning_rate=1e-4, buffer_size=32768, learning_starts=1000, batch_size=32, tau=1.0,
                 gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=10000, exploration_fraction=0.1,
                 exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10):
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.exploration_rate = 0.0
        self.q_net = None
        self.q_net_target = None
        self.replay_buffer = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self.eval_env = None
        self.start_time = None
        self._last_obs = None
        self._last_episode_starts = None
        self._n_calls = 0
        self._current_progress_remaining = 1
        self._n_updates = 0
        self.eval_reward_history = {}
        self.ep_info_buffer = deque(maxlen=100)
        self.ep_success_buffer = deque(maxlen=100)
        self._episode_num = 0
        self.logger = configure_logger()
        self._setup_model()

    def _setup_model(self):
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_space, self.action_space)
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.q_net = self._init_network()
        self.q_net_target = self._init_network()
        polyak_update(self.q_net, self.q_net_target, self.tau)
        self.q_net_target.trainable = False

    def _init_network(self):
        # kernel_initializer=tf.keras.initializers.HeNormal()
        l0 = tf.keras.Input(shape=self.observation_space.shape)
        l1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(l0)
        l2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(l1)
        l3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(l2)
        l4 = tf.keras.layers.Flatten()(l3)
        l5 = tf.keras.layers.Dense(units=512, activation='relu')(l4)
        l6 = tf.keras.layers.Dense(units=9, activation=None)(l5)
        network = tf.keras.Model(inputs=l0, outputs=l6)
        network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                        loss=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE))
        network.summary()
        return network

    def _on_step(self):
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net, self.q_net_target, self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, batch_size):
        self.q_net.trainable = True
        self.logger.record("train/learning_rate", K.eval(self.q_net.optimizer.lr))

        replay_data = self.replay_buffer.sample(batch_size)

        target_q_values = tf.reduce_max(self.q_net_target(replay_data.next_observations), axis=1)
        target_q_values = tf.expand_dims(replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values, axis=-1)

        with tf.GradientTape() as tape:
            current_q_values = tf.gather(self.q_net(replay_data.observations), replay_data.actions, batch_dims=1)
            loss = self.q_net.loss(target_q_values, current_q_values)
        self._update_network(self.q_net, tape, loss)
        self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", K.eval(loss))

    def predict(self, observation):
        if np.random.rand() < self.exploration_rate:
            action = np.array(self.action_space.sample())
        else:
            action = self.q_net(observation)
        return action

    def learn(self, total_timesteps, eval_freq=10, n_eval_episodes=5):
        self.start_time = time.time_ns()
        self._total_timesteps = total_timesteps
        index = 0
        state_init = self.env.reset()
        while self.num_timesteps < total_timesteps:
            state_init = self.collect_rollouts(state_init, 4)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train(batch_size=self.batch_size)
            if self.num_timesteps % eval_freq == 0:
                self.test(n_eval_episodes, index)
                index += 1
        return self

    def _update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        network_gradient, _ = tf.clip_by_global_norm(network_gradient, self.max_grad_norm)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def act(self, observation, test):
        if np.random.random() < self.exploration_schedule(self.num_timesteps) and not test:
            actions = [self.action_space.sample()]
        else:
            state = tf.expand_dims(tf.convert_to_tensor(observation), axis=0)
            actions = [np.argmax(tensor, axis=-1) for tensor in self.q_net(state)]

        return np.array(actions)

    def _update_info_buffer(self, info, done):
        maybe_ep_info = info.get("episode")
        maybe_is_success = info.get("is_success")
        if maybe_ep_info is not None:
            self.ep_info_buffer.extend([maybe_ep_info])
        if maybe_is_success is not None and done:
            self.ep_success_buffer.append(maybe_is_success)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def collect_rollouts(self, state_init, log_interval):
        collected = 0
        while collected < self.train_freq:
            action = self.act(state_init, False)
            state_next, reward, done, info = self.env.step(action[0])
            self.replay_buffer.add(state_init, state_next, action, reward, done)
            collected += 1
            self.num_timesteps += 1
            state_init = state_next
            self._update_info_buffer(info, done)
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()
            if done:
                self._episode_num += 1
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                state_init = self.env.reset()
            '''
            state_next, reward, done, info, render, state_next_adv, reward_adv, done_adv, info_adv = self.env.step(
                action[0])
            collected += 1
            self.num_timesteps += 1
            if done:
                self.replay_buffer.add(state_init, state_next, action, reward, done)
                state_init = self.env.reset()
                break
            elif done_adv:
                self.replay_buffer.add(state_init, state_next_adv, action, reward_adv, done_adv)
                state_init = self.env.reset()
                break
            self.replay_buffer.add(state_init, state_next_adv, action, reward_adv, done_adv)
            state_init = state_next_adv
            '''
        return state_init

    def test(self, n_eval_episodes, index):
        scores = self.play_test_games(n_eval_episodes)
        self.eval_reward_history[index] = scores
        self.display_stats(scores, n_eval_episodes)

    def play_test_games(self, n_eval_episodes):
        scores = []
        for _ in range(n_eval_episodes):
            agent_first = np.random.choice([True, False])
            scores.append(self.play_full_game(agent_first))
        return scores

    def play_full_game(self, agent_first):
        test_env = TicTacToeEnv('Graphic', agent_first)
        test_env = OpponentWrapper2(test_env, 'Random', None)
        state_init = test_env.reset()
        game_frame = 0
        score = 0
        done = False
        while not done:
            action = self.act(state_init, True)
            state_next, reward, done, info = test_env.step(
                action[0])
            game_frame += 2
            score += reward
            state_init = state_next
            '''
            state_next, reward, done, info, render, state_next_adv, reward_adv, done_adv, info_adv = test_env.step(
                action[0])
            game_frame.append(render)
            if done:
                score += reward
            elif done_adv:
                game_frame.append(test_env.render_board(test_env.get_fixed_obs()))
                score += reward_adv
            else:
                state_init = state_next_adv
            '''
        return score, game_frame

    def display_stats(self, results, n_eval_episodes):
        text = 'Timestep : {}\nLearn Steps : {}\nTest Results:\nAverage Score: {:.2f}\nAverage Game Length: {:.2f}' \
               '\nWins: {} | Losses: {} | Ties: {} | Invalid: {}' \
               '\nTest Running Average:\nRunning Average Score: {:.2f}\nRunning Average Game Length: {:.2f}'. \
            format(self.num_timesteps, self._n_updates,
                   sum(i for i, _ in results) / n_eval_episodes,
                   sum(j for _, j in results) / n_eval_episodes,
                   sum(1 for i, _ in results if i == 1), sum(1 for i, _ in results if i == -1),
                   sum(1 for i, _ in results if i == 0), sum(1 for i, _ in results if i == -2),
                   sum(i for t in
                       list(self.eval_reward_history.values())[-100 // n_eval_episodes:]
                       for i, j in t) / (
                       min(len(self.eval_reward_history) * n_eval_episodes, 100)),
                   sum(j for t in
                       list(self.eval_reward_history.values())[-100 // n_eval_episodes:]
                       for i, j in t) / (
                       min(len(self.eval_reward_history) * n_eval_episodes, 100)))
        lines = text.splitlines()
        width = max(len(s) for s in lines)
        res = ['┌' + '─' * (width + 2) + '┐']
        for s in lines:
            res.append('│ ' + (s + ' ' * width)[:width] + ' │')
        res.append('└' + '─' * (width + 2) + '┘')
        print(colored('\n'.join(res), 'magenta'))

    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


def safe_mean(arr):
    return np.nan if len(arr) == 0 else np.mean(arr)


def polyak_update(network, target_network, tau):
    weights = [w * tau + target_network.weights[i] * (1 - tau)
               for i, w in enumerate(network.weights)]
    target_network.set_weights(weights)


def get_linear_fn(start, end, end_fraction):
    def func(progress_remaining):
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


class OpponentWrapper2(Wrapper):
    def __init__(self, env, agent_type, turn):
        super().__init__(env)
        self.agent_type = agent_type
        self.turn = turn
        if self.turn is None:
            self.env.agent_first = np.random.choice([True, False])
        else:
            self.env.agent_first = self.turn

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            return obs, reward, done, {'info': info}
        obs_adv, reward_adv, done_adv, info_adv = self.env.step(self.get_opponent_action())
        return obs_adv, reward_adv, done_adv, {'info': info_adv}

    def get_opponent_action(self):
        if self.agent_type == 'Random':
            valid_actions = self.env.actions(self.env.state)
            return valid_actions[np.random.randint(0, len(valid_actions))]
        elif self.agent_type == 'MinMax':
            return self.env.minmax(self.env.state)
        elif self.agent_type == 'MinMaxRandom':
            return self.env.minmaxran(self.env.state)
        else:
            raise ValueError('Opponent provided does not exist!')

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.turn is None:
            self.env.agent_first = np.random.choice([True, False])
        else:
            self.env.agent_first = self.turn
        if not self.env.agent_first:
            obs, _, _, _ = self.env.step(self.get_opponent_action())
        return obs


class Monitor(Wrapper):
    EXT = "monitor.csv"

    def __init__(self, env):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        self.rewards = None
        self.needs_reset = True
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs):
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_returns

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


if __name__ == '__main__':
    env = Monitor(OpponentWrapper2(TicTacToeEnv('Graphic', None), 'Random', None))
    agent = DQN(env)
    agent.learn(total_timesteps=20000)
    '''
    start_state = agent.replay_buffer.observations[100]
    im1 = Image.fromarray(np.squeeze(start_state)*255.)
    im1.show()
    action = agent.replay_buffer.actions[100]
    print(action)
    end_state = agent.replay_buffer.next_observations[100]
    im2 = Image.fromarray(np.squeeze(end_state)*255.)
    im2.show()
    '''
