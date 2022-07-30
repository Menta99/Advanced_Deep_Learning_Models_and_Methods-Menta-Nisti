import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw

ROWS = 6
COLUMNS = 7
ACTION_SPACE = COLUMNS
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}


class ConnectFourEnv(gym.Env):
    def __init__(self, representation, agent_first):
        self.name = "ConnectFour"
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(COLUMNS, ROWS, 1), dtype=np.int32)
        self.start_mark = 'X'
        self.state = np.zeros((COLUMNS, ROWS))
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic']
        self.done = False
        self.reset()

    def _get_observation(self):
        obs = self.get_fixed_obs()
        if self.representation == 'Tabular':
            return obs
        else:
            return np.asarray(self.render_board(obs))

    def get_fixed_obs(self):
        obs = self.to_image(self.state)
        if not self.agent_first:
            obs = -obs
        return obs

    def to_image(self, state):
        return np.expand_dims(state, axis=-1)

    def reset(self):
        self.start_mark = 'X'
        self.state = np.zeros((COLUMNS, ROWS))
        self.done = False
        return self._get_observation()

    # OpenAI Gym Environments standard function which returns next state given the action to perform, as well as the state of the game (Terminal/non Terminal), action reward and additional informations
    def step(self, action):
        invalidAction = False
        if not self.check_valid_action(self.state, action):
            invalidAction = True
        if invalidAction:
            reward = -2 * self._get_mark()
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, True, 'invalid_action_error'
        else:
            self.result(self.state, action)
        reward, done, info = self.goal(self.state)
        if not self.agent_first:
            reward = -reward
        return self._get_observation(), reward, done, info

    # Returns True if the action is valid, else False
    def check_valid_action(self, state, action):
        return True if state[action][ROWS - 1] == 0 else False

    # Returns all possible actions given the state
    def actions(self, state):
        to_return = []
        for col in range(COLUMNS):
            if self.check_valid_action(state, col):
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
        state[action][sum(map(abs, state[action])).astype('int')] = self._get_mark()
        return state

    # Undo the last action
    def undo(self, state, action):
        state[action][sum(map(abs, state[action])).astype('int') - 1] = 0
        return state

    # Gets current player/symbol by looking at the state of the game (Implicitly 'X' is the first player)
    def _get_mark(self):
        x_counter, o_counter = 0, 0
        for i in range(COLUMNS):
            for j in range(ROWS):
                if self.state[i][j] != 0:
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
                if 4 > i >= 0 and 3 > j >= 0:
                    if (state[i][j] != 0 and state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][
                        j + 3]):
                        return True
                if 3 <= i < COLUMNS and 3 <= j < ROWS:
                    if state[i][j] != 0 and state[i][j] == state[i - 1][j - 1] == state[i - 2][j - 2] == state[i - 3][
                        j - 3]:
                        return True
                if COLUMNS > i >= 3 > j >= 0:
                    if (state[i][j] != 0 and state[i][j] == state[i - 1][j + 1] == state[i - 2][j + 2] == state[i - 3][
                        j + 3]):
                        return True
                if 4 > i >= 0 and 3 <= j < ROWS:
                    if (state[i][j] != 0 and state[i][j] == state[i + 1][j - 1] == state[i + 2][j - 2] == state[i + 3][
                        j - 3]):
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

    def minmaxran(self, state, depth=3):
        alpha = float('-inf')
        beta = float('inf')
        if self.goal(state)[1]:
            return None
        else:
            if self._get_mark() == 1:
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
            aux, act = self.min_value_ran(state, depth-1)
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
            aux, act = self.max_value_ran(state, depth-1)
            if aux < v:
                v = aux
                move = action
                moves = []
            state = self.undo(state, action)  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def render_board(self, state):
        w, h = 224, 192
        image = Image.new('L', (w, h), color=128)
        draw = ImageDraw.Draw(image)
        for i in range(COLUMNS):
            for j in range(ROWS):
                if state[i][j] != 0:
                    if state[i][j] == 1:
                        image.paste(Image.new('L', (32, 32), color=255), (32 * i, -32 * j + 192 - 32))
                    else:
                        image.paste(Image.new('L', (32, 32), color=0), (32 * i, -32 * j + 192 - 32))
        return image
