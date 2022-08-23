import gym
from gym import spaces
import numpy as np
from PIL import Image

ROWS = 6
COLUMNS = 7
ACTION_SPACE = COLUMNS
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}
WIDTH = 224
HEIGHT = 192


class ConnectFourEnv(gym.Env):
    def __init__(self, representation, agent_first):
        self.name = "ConnectFour"
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic'] and self.agent_first in [True, False, None]
        self.action_space = spaces.Discrete(ACTION_SPACE)
        if self.representation == 'Tabular':
            self.observation_space = spaces.Box(low=0., high=1., shape=(ROWS, COLUMNS, 1), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0., high=1., shape=(HEIGHT, WIDTH, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.state = np.zeros((COLUMNS, ROWS))
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic']
        self.done = False
        self.reset()

    def _get_observation(self):
        return self.normalize_obs(self.get_fixed_obs())

    def normalize_obs(self, obs):
        if self.representation == 'Tabular':
            return (np.array(obs, dtype=np.float32) + 1.) / 2.
        else:
            return np.expand_dims(np.asarray(self.render_board(obs), dtype=np.float32), axis=-1) / 255.

    def get_fixed_obs(self):
        obs = self.to_image(self.state)
        if not self.agent_first:
            obs = -obs
        return obs

    def to_image(self, state):
        return np.expand_dims(np.flip(state, axis=1).T, axis=-1)

    def reset(self):
        self.state = np.zeros((COLUMNS, ROWS))
        self.done = False
        return self._get_observation()

    # OpenAI Gym Environments standard function which returns next state given the action to perform, as well as the state of the game (Terminal/non Terminal), action reward and additional informations
    def step(self, action):
        invalidAction = False
        if not self.check_valid_action(self.state, action):
            invalidAction = True
        if invalidAction:
            reward = -2 * self._get_mark(self.state)
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, True, 'invalid_action_error'
        else:
            self.state = self.result(self.state, action)
        reward, done, info = self.goal(self.state)
        if not self.agent_first:
            reward = -reward
        return self._get_observation(), reward, done, info

    # Returns True if the action is valid, else False
    def check_valid_action(self, state, action):
        return True if state[action][ROWS - 1] == 0 else False

    # Returns all possible actions given the state
    def actions(self, state):
        return [col for col in range(COLUMNS) if self.check_valid_action(state, col)]

    # Checks whether a final state is reached
    def goal(self, state):
        if self._check_diagonal(state) or self._check_horizontal(state) or self._check_vertical(state):
            if self._get_mark(state) == -1:
                return X_REWARD, True, "X won"
            else:
                return O_REWARD, True, "O won"
        elif len(self.actions(state)) == 0:
            return TIE_REWARD, True, "Tie"
        else:
            return 0, False, 'Game not End'

    # Returns Next state given current state and action
    def result(self, state, action):
        state_copy = state.copy()
        state_copy[action][sum(map(abs, state_copy[action])).astype('int')] = 1 if np.count_nonzero(state_copy == 1) == np.count_nonzero(state_copy == -1) else -1
        return state_copy

    # Undo the last action
    def undo(self, state, action):
        state[action][sum(map(abs, state[action])).astype('int') - 1] = 0
        return state

    # Gets current player/symbol by looking at the state of the game (Implicitly 'X' is the first player)
    def _get_mark(self, state):
        return 1 if np.count_nonzero(state == 1) == np.count_nonzero(state == -1) else -1

    # Checks winning conditions -> 4 of the same symbol in a row, a column or diagonally
    def _check_horizontal(self, state):
        for i in range(ROWS - 1, -1, -1):
            for j in range(4):
                if state[j, i] != 0 and state[j, i] == state[j + 1, i] == state[j + 2, i] == state[j + 3, i]:
                    return True
        return False

    def _check_vertical(self, state):
        for i in range(COLUMNS):
            for j in range(2, -1, -1):
                if state[i, j] != 0 and state[i, j] == state[i, j + 1] == state[i, j + 2] == state[i, j + 3]:
                    return True
        return False

    def _check_diagonal(self, state):
        for i in range(4):
            for j in range(2, -1, -1):
                if state[i, j] != 0 and state[i, j] == state[i + 1, j + 1] == state[i + 2, j + 2] == state[i + 3, j + 3]:
                    return True
                if state[i + 3, j] != 0 and state[i + 3, j] == state[i + 2, j + 1] == state[i + 1, j + 2] == state[i, j + 3]:
                    return True
        return False

    ###############################################
    # Alpha Beta Pruning AI to select the best move#
    ###############################################
    def minmax(self, state, depth=6):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self._get_mark(state) == 1:
                value, move = self.max_value(state, alpha, beta, depth)
                return move
            else:
                value, move = self.min_value(state, alpha, beta, depth)
                return move

    def max_value(self, state, alpha, beta, depth):  # Evaluate max node values and their best moves
        if self.goal(state)[1] or depth == 0:  # If Done or maximum depth reached
            return self.goal(state)[0], None  # Return utility/reward
        v = float('-inf')  # Initialize Node value
        move = None
        for action in self.actions(
                state):  # For each possible action evaluate next state value and update node max value
            state = self.result(state, action)
            # v = max(v, min_v(next_state))
            aux, act = self.min_value(state, alpha, beta, depth - 1)
            if aux > v:
                v = aux
                move = action
            state = self.undo(state, action)
            alpha = max(alpha, aux)  # Evaluate Alpha value
            if beta <= alpha:  # Prune branch -> worst than lower bound / (worst case scenario)
                break
        return v, move

    def min_value(self, state, alpha, beta, depth):  # Evaluate min node values and their best moves
        if self.goal(state)[1] or depth == 0:
            return self.goal(state)[0], None

        v = float('inf')
        move = None
        for action in self.actions(state):
            state = self.result(state, action)
            # v = max(v, min_v(next_state))
            aux, act = self.max_value(state, alpha, beta, depth - 1)
            if aux < v:
                v = aux
                move = action
            state = self.undo(state, action)
            beta = min(beta, aux)
            if beta <= alpha:
                break
        return v, move

    def minmaxran(self, state, depth=3):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self._get_mark(state) == 1:
                value, moves = self.max_value_ran(state, depth, True)
                return np.random.choice(moves)
            else:
                value, moves = self.min_value_ran(state, depth, True)
                return np.random.choice(moves)

    def max_value_ran(self, state, depth, save_actions=False):
        if self.goal(state)[1] or depth == 0:
            return self.goal(state)[0], []
        v = float('-inf')
        move = None
        moves = []
        for action in self.actions(state):
            state = self.result(state, action)
            # v = max(v, min_v(next_state))
            aux, act = self.min_value_ran(state, depth - 1)
            if aux > v:
                v = aux
                move = action
                moves = []
            state = self.undo(state, action)  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def min_value_ran(self, state, depth, save_actions=False):
        if self.goal(state)[1] or depth == 0:
            return self.goal(state)[0], []

        v = float('inf')
        move = None
        moves = []
        for action in self.actions(state):
            state = self.result(state, action)
            # v = max(v, min_v(next_state))
            aux, act = self.max_value_ran(state, depth - 1)
            if aux < v:
                v = aux
                move = action
                moves = []
            state = self.undo(state, action)  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def render_board(self, state):
        image = Image.new('L', (WIDTH, HEIGHT), color=128)
        for i in range(ROWS):
            for j in range(COLUMNS):
                if state[i][j] != 0:
                    if state[i][j] == 1:
                        image.paste(Image.new('L', (32, 32), color=255), (32 * j, 32 * i)) #+ HEIGHT + 32
                    else:
                        image.paste(Image.new('L', (32, 32), color=0), (32 * j, 32 * i)) #+ HEIGHT + 32
        return image
