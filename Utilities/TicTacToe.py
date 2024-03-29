from copy import copy

import gym
import numpy as np
from PIL import Image
from gym import spaces

BOARD_SIZE = 3
ACTION_SPACE = 9
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}
WIDTH = 96
HEIGHT = 96


class TicTacToeEnv(gym.Env):
    def __init__(self, representation, agent_first):
        self.name = "TicTacToe"
        self.possible_actions = list(range(9))
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic'] and self.agent_first in [True, False, None]
        self.action_space = spaces.Discrete(ACTION_SPACE)
        if self.representation == 'Tabular':
            self.observation_space = spaces.Box(low=0., high=1., shape=(3, 3, 1), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0., high=1., shape=(WIDTH, HEIGHT, 1), dtype=np.float32)
        self.state = np.zeros((3, 3))  # ACTION_SPACE * [0]
        self.turn = 0
        self.done = False
        self.exploration_parameter = 1
        self.reset()

    def _get_observation(self):
        return self.normalize_obs(self.get_fixed_obs())

    def normalize_obs(self, obs):
        if self.representation == 'Tabular':
            return (np.array(obs, dtype=np.float32) + 1.) / 2.
        else:
            return np.expand_dims(np.asarray(self.render_board(obs), dtype=np.float32), axis=-1) / 255.

    def get_fixed_obs(self):
        obs = self.to_image()
        if not self.agent_first:
            obs = -obs
        return obs

    def to_image(self):
        return np.expand_dims(self.state, axis=-1)

    def reset(self):
        self.state = np.zeros((3, 3))
        self.done = False
        return self._get_observation()

    def step(self, action):
        invalidAction = False
        if self.state[action // 3][action % 3] != 0:
            invalidAction = True
        if invalidAction:
            reward = -2 * self.get_mark(self.state)
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, True, 'invalid_action_error'
        else:
            self.state[action // 3][action % 3] = self.get_mark(self.state)
        self.turn += 1
        a, b, c = self.goal(self.state)
        if not self.agent_first:
            a = -a
        return self._get_observation(), a, b, c

    # Returns all possible actions given the state
    def actions(self, state):
        return [i for i in range(9) if state[i // 3][i % 3] == 0]

    def get_random_action(self, state):
        np.random.shuffle(self.possible_actions)
        for a in self.possible_actions:
            if state[a // 3][a % 3] == 0:
                return a
        return None

    # Checks wheter a final state is reached
    def goal(self, state):
        if self._check_diagonal(state) or self._check_horizontal(state) or self._check_vertical(state):
            if self.get_mark(state) == -1:
                return X_REWARD, True, "X won"
            else:
                return O_REWARD, True, "O won"
        elif len(self.actions(state)) == 0:
            return TIE_REWARD, True, "Tie"
        else:
            return 0, False, 'Game not End'

    # Returns Next state given current state and action
    def result(self, state, action):
        state_copy = copy(state)
        state_copy[action // 3][action % 3] = 1 if np.count_nonzero(state_copy == 1) == np.count_nonzero(state_copy == -1) else -1#self._get_mark()
        return state_copy

    def get_mark(self, state):
        return 1 if np.count_nonzero(state == 1) == np.count_nonzero(state == -1) else -1

    def _check_horizontal(self, state):
        return BOARD_SIZE in abs(np.sum(state, axis=1))

    def _check_vertical(self, state):
        return BOARD_SIZE in abs(np.sum(state, axis=0))

    def _check_diagonal(self, state):
        return state[1][1] != 0 and (
                state[0][0] == state[1][1] == state[2][2] or state[2][0] == state[1][1] == state[0][2])

    # Alpha Beta Pruning AI to select the best move
    def minmax(self, state):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self.get_mark(state) == 1:
                value, move = self.max_value(state, alpha, beta)
                return move
            else:
                value, move = self.min_value(state, alpha, beta)
                return move

    def max_value(self, state, alpha, beta):
        if self.goal(state)[1]:
            return self.goal(state)[0], None
        v = float('-inf')
        move = None
        for action in self.actions(state):
            state[action // 3][action % 3] = self.get_mark(state)
            # v = max(v, min_v(next_state))
            aux, act = self.min_value(state, alpha, beta)
            if aux > v:
                v = aux
                move = action
            state[action // 3][action % 3] = 0  # Undo move
            alpha = max(alpha, aux)
            if beta <= alpha:
                break
        return v, move

    def min_value(self, state, alpha, beta):
        if self.goal(state)[1]:
            return self.goal(state)[0], None
        v = float('inf')
        move = None
        for action in self.actions(state):
            state[action // 3][action % 3] = self.get_mark(state)
            # v = max(v, min_v(next_state))
            aux, act = self.max_value(state, alpha, beta)
            if aux < v:
                v = aux
                move = action
            state[action // 3][action % 3] = 0  # Undo move
            beta = min(beta, aux)
            if beta <= alpha:
                break
        return v, move

    def minmaxran(self, state):
        action = self.minmax(state)
        if action == 4:
            return action
        elif action in [0, 2, 6, 8]:
            if np.equal(np.flip(state, axis=1), state).all() and np.equal(np.flip(state, axis=0), state).all():
                return np.random.choice([0, 2, 6, 8])
            elif np.equal(np.flip(state, axis=1), state).all():
                if action in [0, 2]:
                    return np.random.choice([0, 2])
                else:
                    return np.random.choice([6, 8])
            elif np.equal(np.flip(state, axis=0), state).all():
                if action in [0, 6]:
                    return np.random.choice([0, 6])
                else:
                    return np.random.choice([2, 8])
            elif np.equal(state.T, state).all():
                if action in [2, 6]:
                    return np.random.choice([2, 6])
                else:
                    return action
            elif np.equal(state[::-1, ::-1].T, state).all():
                if action in [0, 8]:
                    return np.random.choice([0, 8])
                else:
                    return action
            else:
                return action
        else:
            if np.equal(np.flip(state, axis=1), state).all() and np.equal(np.flip(state, axis=0), state).all():
                return np.random.choice([1, 3, 5, 7])
            elif np.equal(np.flip(state, axis=1), state).all():
                if action in [3, 5]:
                    return np.random.choice([3, 5])
                else:
                    return action
            elif np.equal(np.flip(state, axis=0), state).all():
                if action in [1, 7]:
                    return np.random.choice([1, 7])
                else:
                    return action
            elif np.equal(state.T, state).all():
                if action in [1, 3]:
                    return np.random.choice([1, 3])
                else:
                    return np.random.choice([5, 7])
            elif np.equal(state[::-1, ::-1].T, state).all():
                if action in [1, 5]:
                    return np.random.choice([1, 5])
                else:
                    return np.random.choice([3, 7])
            else:
                return action

    def render_board(self, state):
        image = Image.new('L', (WIDTH, HEIGHT), color=128)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[j][i] != 0:
                    if state[j][i] == 1:
                        image.paste(Image.new('L', (32, 32), color=255), (32 * i, 32 * j))
                    else:
                        image.paste(Image.new('L', (32, 32), color=0), (32 * i, 32 * j))
        return image
