import gym
from PIL import Image
from gym import spaces
import numpy as np

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
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic'] and self.agent_first in [True, False, None]
        self.action_space = spaces.Discrete(ACTION_SPACE)
        if self.representation == 'Tabular':
            self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 1), dtype=np.int32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 1), dtype=np.int32)
        self.state = np.zeros((3, 3))  # ACTION_SPACE * [0]
        self.turn = 0
        self.done = False
        self.reset()

    def _get_observation(self):
        obs = self.get_fixed_obs()
        if self.representation == 'Tabular':
            return obs
        else:
            return np.expand_dims(np.asarray(self.render_board(obs)), axis=-1)

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
            reward = -2 * self._get_mark()
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, True, 'invalid_action_error'
        else:
            self.state[action // 3][action % 3] = self._get_mark()
        self.turn += 1
        a, b, c = self.goal(self.state)
        if not self.agent_first:
            a = -a
        return self._get_observation(), a, b, c

    # Returns all possible actions given the state
    def actions(self, state):
        return [i for i in range(9) if state[i // 3][i % 3] == 0]

    # Checks wheter a final state is reached
    def goal(self, state):
        if len(self.actions(state)) == 0:
            return TIE_REWARD, True, "Tie"
        elif self._check_diagonal(state) or self._check_horizontal(state) or self._check_vertical(state):
            if self._get_mark() == -1:
                return X_REWARD, True, "X won"
            else:
                return O_REWARD, True, "O won"
        else:
            return 0, False, 'Game not End'

    # Returns Next state given current state and action
    def result(self, state, action):
        state[action // 3][action % 3] = self._get_mark()
        return state

    def _get_mark(self):
        return 1 if np.count_nonzero(self.state == 1) == np.count_nonzero(self.state == -1) else -1

    def _check_horizontal(self, state):
        return BOARD_SIZE in abs(np.sum(state, axis=1))

    def _check_vertical(self, state):
        return BOARD_SIZE in abs(np.sum(state, axis=0))

    def _check_diagonal(self, state):
        return state[1][1] != 0 and (state[0][0] == state[1][1] == state[2][2] or state[2][0] == state[1][1] == state[0][2])

    # Alpha Beta Pruning AI to select the best move
    def minmax(self, state):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self._get_mark() == 1:
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
            state[action // 3][action % 3] = self._get_mark()
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
            state[action // 3][action % 3] = self._get_mark()
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

    # MiMax that returns a random non-suboptimal move
    def minmaxran(self, state):
        if self.goal(state)[1]:
            return None
        else:
            if self.turn == 0:
                return np.random.choice([i for i in range(ACTION_SPACE)])
            if self.turn == 1:
                if state[1][1] != 0:
                    return np.random.choice([0, 2, 6, 8])
                elif state[0][0] != 0 or state[0][2] != 0 or state[2][0] != 0 or state[2][2] != 0:
                    return 4
                else:
                    return np.random.choice(self.actions(state))
            if self.turn == 2:
                act = self._get_mark()
                if state[0][1] == -act or state[1][0] == -act or state[1][2] == -act or state[2][1] == -act:
                    if state[1][1] != 0:
                        return 4
                return np.random.choice(self.actions(state))
            if self._get_mark() == 1:
                value, moves = self.max_value_ran(state, True)
                return np.random.choice(moves)
            else:
                value, moves = self.min_value_ran(state, True)
                return np.random.choice(moves)

    def max_value_ran(self, state, save_actions=False):
        if self.goal(state)[1]:
            return self.goal(state)[0], []
        v = float('-inf')
        moves = []
        for action in self.actions(state):
            state[action // 3][action % 3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.min_value_ran(state)
            if aux > v:
                v = aux
                moves = []
            state[action // 3][action % 3] = 0  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def min_value_ran(self, state, save_actions=False):
        if self.goal(state)[1]:
            return self.goal(state)[0], []
        v = float('inf')
        moves = []
        for action in self.actions(state):
            state[action // 3][action % 3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.max_value_ran(state)
            if aux < v:
                v = aux
                moves = []
            state[action // 3][action % 3] = 0  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

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
