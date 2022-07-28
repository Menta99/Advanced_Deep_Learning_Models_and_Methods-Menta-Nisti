import gym
from gym import spaces
import numpy as np

ROWS = 6
COLUMNS = 7
ACTION_SPACE = COLUMNS
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}


class ConnectFourEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Discrete(ROWS * COLUMNS)
        self.start_mark = 'X'
        self.state = [[0 for i in range(ROWS)] for j in range(COLUMNS)]
        self.done = False
        self.reset()

    def _get_observation(self):
        return self.to_image(self.state)

    def to_image(self, state):
        return np.array([state[i] for i in range(COLUMNS)])

    def reset(self):
        self.start_mark = 'X'
        self.state = [[0 for i in range(ROWS)] for j in range(COLUMNS)]
        self.done = False
        return self._get_observation()

    # OpenAI Gym Environments standard function which returns next state given the action to perform, as well as the state of the game (Terminal/non Terminal), action reward and additional informations
    def step(self, action):
        invalidAction = False
        if (not self.check_valid_action(action)):
            invalidAction = True
        if invalidAction:
            return self._get_observation(), -2 * self._get_mark(), True, 'invalid_action_error'  # Invalid Action Reward = -1 for X, 1 for O
        else:
            self.result(self.state, action)
        reward, done, info = self.goal(self.state)
        return self._get_observation(), reward, done, info

    # Returns True if the action is valid, else False
    def check_valid_action(self, action):
        return True if self.state[action][ROWS - 1] == 0 else False

    # Returns all possible actions given the state
    def actions(self, state):
        to_return = []
        for col in range(COLUMNS):
            if (self.check_valid_action(col)):
                to_return.append(col)
        return to_return

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
        state[action][len_occupied(state[action])] = self._get_mark()
        return state

    # Undo the last action
    def undo(self, state, action):
        state[action][len_occupied(state[action]) - 1] = 0
        return state

    # Gets current player/symbol by looking at the state of the game (Implicitly 'X' is the first player)
    def _get_mark(self):
        x_counter, o_counter = 0, 0
        for i in range(COLUMNS):
            for j in range(ROWS):
                if (self.state[i][j] != 0):
                    if self.state[i][j] == 1:
                        x_counter += 1
                    else:
                        o_counter += 1

        return 1 if x_counter == o_counter else -1

    # Checks winning conditions -> 4 of the same symbol in a row, a column or diagonally
    def _check_horizontal(self, state):
        for i in range(0, ROWS):
            cnt = 0
            for j in range(0, COLUMNS - 1):
                cnt = cnt + 1 if (state[j][i] == state[j + 1][i] and state[j][i] != 0) else 0
                if cnt == 3:
                    return True
        return False

    def _check_vertical(self, state):
        for i in range(0, COLUMNS):
            cnt = 0
            for j in range(0, ROWS - 1):
                cnt = cnt + 1 if (state[i][j] == state[i][j + 1] and state[i][j] != 0) else 0
                if cnt == 3:
                    return True
        return False

    def _check_diagonal(self, state):
        for i in range(COLUMNS):
            for j in range(ROWS):
                if (i < 4 and i >= 0 and j < 3 and j >= 0):
                    if (state[i][j] != 0 and state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][
                        j + 3]):
                        return True
                if i >= 3 and i < COLUMNS and j >= 3 and j < ROWS:
                    if state[i][j] != 0 and state[i][j] == state[i - 1][j - 1] == state[i - 2][j - 2] == state[i - 3][
                        j - 3]:
                        return True
                if (i >= 3 and i < COLUMNS and j < 3 and j >= 0):
                    if (state[i][j] != 0 and state[i][j] == state[i - 1][j + 1] == state[i - 2][j + 2] == state[i - 3][
                        j + 3]):
                        return True
                if (i < 4 and i >= 0 and j >= 3 and j < ROWS):
                    if (state[i][j] != 0 and state[i][j] == state[i + 1][j - 1] == state[i + 2][j - 2] == state[i + 3][
                        j - 3]):
                        return True
        return False

    ###############################################
    # Alpha Beta Pruning AI to select the best move#
    ###############################################
    def minimax(self, state, depth=6):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self._get_mark() == 1:
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


def len_occupied(vector):
    counter = 0
    for i in range(len(vector)):
        if vector[i] != 0:
            counter += 1
    return counter
