import gym
from gym import spaces
import numpy as np

BOARD_SIZE = 3
ACTION_SPACE = 9
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 1), dtype=np.int32)
        self.start_mark = 'X'
        self.state = ACTION_SPACE * [0]
        self.reset()

    def _get_observation(self):
        return self.to_image(self.state)

    def to_image(self, state):
        return np.expand_dims(np.array([state[:3], state[3:6], state[6:9]]), axis=-1)

    def reset(self):
        self.start_mark = 'X'
        self.state = ACTION_SPACE * [0]
        self.done = False
        return self._get_observation()

    def step(self, action):
        invalidAction = False
        if (self.state[action] != 0):
            invalidAction = True
        if invalidAction:
            return self._get_observation(), -2 * self._get_mark(), True, 'invalid_action_error'  # Invalid Action Reward = -1 for X, 1 for O
        else:
            self.state[action] = self._get_mark()
        a, b, c = self.goal(self.state)
        return self._get_observation(), a, b, c

    # Returns all possible actions given the state
    def actions(self, state):
        return [i for i, s in enumerate(state) if s == 0]

    # Checks wheter a final state is reached
    def goal(self, state):
        done = False
        reward = 0
        info = "Game not End"
        win = self._check_diagonal(state) or self._check_horizontal(state) or self._check_vertical(state)
        if win:
            done = True
            reward = X_REWARD if self._get_mark() == -1 else O_REWARD
            info = "X won" if reward == X_REWARD else "O Won"
        tie = len(self.actions(state)) == 0
        if tie and not win:
            done = True
            reward = TIE_REWARD
            info = "tie"
        return reward, done, info

    # Returns Next state given current state and action
    def result(self, state, action):
        state[action] = self._get_mark()
        return state

    # Utility to print the gameboard
    def print_board(self):
        for i in range(0, BOARD_SIZE ** 2, BOARD_SIZE):
            print(SYMBOLS_DICT[self.state[0 + i]] + SYMBOLS_DICT[self.state[1 + i]] + SYMBOLS_DICT[self.state[2 + i]])

    def _get_mark(self):
        x_counter, o_counter = 0, 0
        for i in range(BOARD_SIZE ** 2):
            if (self.state[i] != 0):
                if self.state[i] == 1:
                    x_counter += 1
                else:
                    o_counter += 1

        return 1 if x_counter == o_counter else -1

    def _check_horizontal(self, state):
        for i in range(0, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE):
            cnt = 0
            k = i
            for j in range(1, BOARD_SIZE):
                (cnt, k) = (cnt + 1, k) if (state[k] == state[i + j] and state[k] != 0) else (0, i + j)
            if cnt == BOARD_SIZE - 1:
                return True
        return False

    def _check_vertical(self, state):
        for i in range(0, BOARD_SIZE):
            cnt = 0
            k = i
            for j in range(BOARD_SIZE, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE):
                (cnt, k) = (cnt + 1, k) if (state[k] == state[i + j] and state[k] != 0) else (0, i + j)
                if cnt == BOARD_SIZE - 1:
                    return True
        return False

    def _check_diagonal(self, state):
        if ((state[0] == state[4] == state[8] or state[2] == state[4] == state[6]) and state[4] != 0):
            return True
        return False

    # Alpha Beta Pruning AI to select the best move
    def minimax(self, state):
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
            state[action] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.min_value(state, alpha, beta)
            if aux > v:
                v = aux
                move = action
            state[action] = 0  # Undo move
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
            state[action] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.max_value(state, alpha, beta)
            if aux < v:
                v = aux
                move = action
            state[action] = 0  # Undo move
            beta = min(beta, aux)
            if beta <= alpha:
                break
        return v, move

    # MiMax that returns a random non-suboptimal move
    def minimaxran(self, state):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if state == ACTION_SPACE * [0]:
                return np.random.choice([i for i in range(ACTION_SPACE)])
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
        move = None
        moves = []
        for action in self.actions(state):
            state[action] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.min_value_ran(state)
            if aux > v:
                v = aux
                move = action
                moves = []
            state[action] = 0  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def min_value_ran(self, state, save_actions=False):
        if self.goal(state)[1]:
            return self.goal(state)[0], []

        v = float('inf')
        move = None
        moves = []
        for action in self.actions(state):
            state[action] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.max_value_ran(state)
            if aux < v:
                v = aux
                move = action
                moves = []
            state[action] = 0  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves
