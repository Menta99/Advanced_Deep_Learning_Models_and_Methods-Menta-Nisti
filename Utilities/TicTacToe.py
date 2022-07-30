import gym
from PIL import Image, ImageDraw
from gym import spaces
import numpy as np

BOARD_SIZE = 3
ACTION_SPACE = 9
X_REWARD = 1
O_REWARD = -1
TIE_REWARD = 0
SYMBOLS_DICT = {0: '_', 1: 'X', -1: 'O'}


class TicTacToeEnv(gym.Env):
    def __init__(self, representation, agent_first):
        self.name = "TicTacToe"
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 1), dtype=np.int32)
        self.start_mark = 'X'
        self.state = np.zeros((3,3)) #ACTION_SPACE * [0]
        self.turn = 0
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
        return np.expand_dims(self.state, axis=-1)

    def reset(self):
        self.start_mark = 'X'
        self.state = np.zeros((3,3))
        self.done = False
        return self._get_observation()

    def step(self, action):
        invalidAction = False
        if self.state[action//3][action%3] != 0:
            invalidAction = True
        if invalidAction:
            reward = -2 * self._get_mark()
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, True, 'invalid_action_error'
        else:
            self.state[action//3][action%3] = self._get_mark()
        self.turn += 1
        a, b, c = self.goal(self.state)
        if not self.agent_first:
            a = -a
        return self._get_observation(), a, b, c

    # Returns all possible actions given the state
    def actions(self, state):
        actions = []
        for i in range(9):
          if state[i//3][i%3] == 0:
            actions.append(i)
        return actions

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
        state[action//3][action%3] = self._get_mark()
        return state

    def _get_mark(self):
        x_counter, o_counter = 0, 0
        for i in range(BOARD_SIZE ** 2):
            if (self.state[i//3][i%3] != 0):
                if self.state[i//3][i%3] == 1:
                    x_counter += 1
                else:
                    o_counter += 1

        return 1 if x_counter == o_counter else -1

    def _check_horizontal(self, state):
        for i in range(0, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE):
            cnt = 0
            k = i
            for j in range(1, BOARD_SIZE):
                (cnt, k) = (cnt + 1, k) if (state[k//3][k%3] == state[(i + j)//3][(i+j)%3] and state[k//3][k%3] != 0) else (0, i + j)
            if cnt == BOARD_SIZE - 1:
                return True
        return False

    def _check_vertical(self, state):
        for i in range(0, BOARD_SIZE):
            cnt = 0
            k = i
            for j in range(BOARD_SIZE, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE):
                (cnt, k) = (cnt + 1, k) if (state[k//3][k%3] == state[(i + j)//3][(i+j)%3] and state[k//3][k%3] != 0) else (0, i + j)
                if cnt == BOARD_SIZE - 1:
                    return True
        return False

    def _check_diagonal(self, state):
        if ((state[0][0] == state[1][1] == state[2][2] or state[2][0] == state[1][1] == state[0][2]) and state[1][1] != 0):
            return True
        return False

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
            state[action//3][action%3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.min_value(state, alpha, beta)
            if aux > v:
                v = aux
                move = action
            state[action//3][action%3] = 0  # Undo move
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
            state[action//3][action%3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.max_value(state, alpha, beta)
            if aux < v:
                v = aux
                move = action
            state[action//3][action%3] = 0  # Undo move
            beta = min(beta, aux)
            if beta <= alpha:
                break
        return v, move

    # MiMax that returns a random non-suboptimal move
    def minmaxran(self, state):
        alpha = float('-inf')
        beta = float('inf')
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
        move = None
        moves = []
        for action in self.actions(state):
            state[action//3][action%3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.min_value_ran(state)
            if aux > v:
                v = aux
                move = action
                moves = []
            state[action//3][action%3] = 0  # Undo move
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
            state[action//3][action%3] = self._get_mark()
            # v = max(v, min_v(next_state))
            aux, act = self.max_value_ran(state)
            if aux < v:
                v = aux
                move = action
                moves = []
            state[action//3][action%3] = 0  # Undo move
            if save_actions and aux == v:
                moves.append(action)
        return v, moves

    def render_board(self, state):
        w, h = 96, 96
        image = Image.new('L', (w, h), color=128)
        draw = ImageDraw.Draw(image)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[j][i] != 0:
                    if state[j][i] == 1:
                        image.paste(Image.new('L', (32, 32), color=255), (32 * i, 32 * j))
                    else:
                        image.paste(Image.new('L', (32, 32), color=0), (32 * i, 32 * j))
        return image
