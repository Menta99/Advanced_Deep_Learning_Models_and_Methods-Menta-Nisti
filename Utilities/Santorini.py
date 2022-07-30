import itertools
import gym
from gym import spaces
import numpy as np
from PIL import Image, ImageDraw

BOARD_SIZE = 5

ACTION_SPACE = np.array([2, 3, 3, 3, 3, 5, 5])
ONE_REWARD = 1
TWO_REWARD = -1
TIE_REWARD = 0
LAYERS = {'first': 0,
          'second': 1,
          'third': 2,
          'dome': 3,
          'player1': 4,
          'player2': 5
          }

ACTIONS = {
    0: [1, -1],
    1: [1, 0],
    2: [1, 1],
    3: [0, -1],
    4: [0, 1],
    5: [-1, -1],
    6: [-1, 0],
    7: [-1, 1]
}


class SantoriniEnv(gym.Env):
    def __init__(self, representation, agent_first, random_init=True):
        self.name = "Santorini"
        self.action_space = spaces.MultiDiscrete(ACTION_SPACE)
        self.observation_space = spaces.Box(low=0, high=1, shape=(BOARD_SIZE, BOARD_SIZE, len(LAYERS), 1),
                                            dtype=np.int32)
        self.player_one = True
        self.turn = 0
        self.random_init = random_init
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, len(LAYERS)))
        self.player_one_workers = [[], []]
        self.player_two_workers = [[], []]
        self.representation = representation
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic']
        self.done = False
        self.reset()

    def reset(self):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, len(LAYERS)))
        self.player_one = True
        self.turn = 0
        self.done = False

        if self.random_init:
            self.turn = 4

            for i in range(len(self.player_one_workers)):
                self.player_one_workers[i] = self._assign_worker(0)
                self.player_two_workers[i] = self._assign_worker(1)

        return self._get_observation()

    def _get_observation(self):
        obs = self.get_fixed_obs()
        if self.representation == 'Tabular':
            return obs
        else:
            return np.expand_dims(np.asarray(self.render_board(obs)), axis=-1)

    def get_fixed_obs(self):
        obs = np.expand_dims(self.state, axis=-1)
        if not self.agent_first:
            obs = np.concatenate([obs[:, :, 1:2, :], obs[:, :, 0:1, :], obs[:, :, 2:, :]], axis=-2)
        return obs

    def _assign_worker(self, player_num, action=None):
        if self.random_init:
            coord = [np.random.choice(range(BOARD_SIZE)), np.random.choice(range(BOARD_SIZE))]
            if all(self.state[coord[0]][coord[1]][LAYERS['player1']:] == 0):
                self.state[coord[0]][coord[1]][LAYERS['player1'] + player_num] = 1
                return coord
            else:
                return self._assign_worker(player_num)
        elif action is not None:
            if all(self.state[action[5]][action[6]][LAYERS['player1']:] == 0):
                self.state[action[5]][action[6]][LAYERS['player1'] + player_num] = 1
                return [action[5], action[6]], False
            else:
                return [], True
        return

    # OpenAI Gym Environments standard function which returns next state given the action to perform, as well as the state of the game (Terminal/non Terminal), action reward and additional informations
    def step(self, action):

        if len(action) == 3:  # 2,8,8 ---> 2, 3,3, 3,3
            action = expand_action(action)

        player_num = 0 if self.player_one else 1

        if self.turn < 4:
            coordinates, wrong_init = self._assign_worker(player_num, action)

            if wrong_init:
                reward = TWO_REWARD if player_num == 0 else ONE_REWARD
                if not self.agent_first:
                    reward = -reward
                done = True
                info = "Wrong initialization by player 1" if player_num == 0 else "Wrong initialization by player 2"
                return self._get_observation(), reward, done, info

            if self.turn % 2 == 0:
                self.player_one_workers[self.turn // 2] = coordinates
            else:
                self.player_two_workers[self.turn // 2] = coordinates
            self.turn += 1
            self.player_one = not self.player_one
            return self._get_observation(), 0, False, "Initialization phase"

        if action is None:
            reward = TWO_REWARD if player_num == 0 else ONE_REWARD
            info = "player 1 has no more moves, player 2 wins" if player_num == 0 else "player 2 has no more moves, player 1 wins"
            done = True
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, done, info

        if not self.check_valid_action(action):
            reward = -2 * ONE_REWARD if player_num == 0 else -2 * TWO_REWARD
            info = "Illegal action by player 1" if player_num == 0 else "Illegal action by player 2"
            done = True
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, done, info

        coord_worker = self.player_one_workers[action[0]] if self.player_one else self.player_two_workers[action[0]]

        self.state[coord_worker[0]][coord_worker[1]][LAYERS['player1'] + player_num] = 0
        self.state[coord_worker[0] + action[1]][coord_worker[1] + action[2]][LAYERS['player1'] + player_num] = 1

        if self.player_one:
            self.player_one_workers[action[0]] = [coord_worker[0] + action[1], coord_worker[1] + action[2]]
        else:
            self.player_two_workers[action[0]] = [coord_worker[0] + action[1], coord_worker[1] + action[2]]

        coord_worker = self.player_one_workers[action[0]] if self.player_one else self.player_two_workers[action[0]]

        self._build(coord_worker[0] + action[3], coord_worker[1] + action[4])

        self.player_one = not self.player_one
        self.turn += 1

        reward, done, info = self.goal()
        if not self.agent_first:
            reward = -reward
        return self._get_observation(), reward, done, info

    def _build(self, build_row, build_column):
        level = 0
        while self.state[build_row][build_column][level] == 1 and level < 3:
            level += 1
        self.state[build_row][build_column][level] = 1
        return

    # Returns True if the action is valid, else False
    def check_valid_action(self, action):
        if len(action) == 3:  # 2,8,8 ---> 2, 3,3, 3,3
            action = expand_action(action)

        if action[1] == action[2] == 0 or action[3] == action[4] == 0:
            return False  # invalid action no movement/build on spot

        coord_worker = self.player_one_workers[action[0]] if self.player_one else self.player_two_workers[action[0]]
        coord_landing = [coord_worker[0] + action[1], coord_worker[1] + action[2]]
        coord_building = [coord_landing[0] + action[3], coord_landing[1] + action[4]]

        if any(i > 4 for i in coord_landing) or any(i < 0 for i in coord_landing):
            return False  # move out of list

        if any(i > 4 for i in coord_building) or any(i < 0 for i in coord_building):
            return False  # building out of list

        if any(self.state[coord_landing[0]][coord_landing[1]][3:]) == 1:
            return False  # no move for occupied or dome

        if any(self.state[coord_building[0]][coord_building[1]][3:]) == 1:
            return False  # no building for occupied or dome

        if sum(self.state[coord_landing[0]][coord_landing[1]][:3]) > sum(
                self.state[coord_worker[0]][coord_worker[1]][:3]) + 1:
            return False  # too high destination

        return True

    # Returns all possible actions given the state in the expanded form
    def actions(self, state):
        actions = []
        for action in itertools.product(*[[0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]):
            action = list(action)
            if self.check_valid_action(action):
                actions.append(action)
        return actions

    # Checks whether a final state is reached
    def goal(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.state[i][j][LAYERS['third']] == 1:
                    if self.state[i][j][LAYERS['player1']] == 1:
                        return ONE_REWARD, True, "player one wins"
                    elif self.state[i][j][LAYERS['player2']] == 1:
                        return TWO_REWARD, True, "player two wins"
        return 0, False, "game not end"

    def render_board(self, state):
        w, h = 160, 160
        image = Image.new('L', (w, h))
        draw = ImageDraw.Draw(image)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i][j][0] == 1:
                    image.paste(Image.new('L', (32, 32), color=255 // 5), (32 * i, 32 * j))
                if state[i][j][1] == 1:
                    image.paste(Image.new('L', (24, 24), color=255 // 5 * 2), (32 * i + 4, 32 * j + 4))
                if state[i][j][2] == 1:
                    image.paste(Image.new('L', (16, 16), color=255 // 5 * 3), (32 * i + 8, 32 * j + 8))
                if state[i][j][3] == 1:
                    image.paste(Image.new('L', (8, 8), color=255 // 5 * 4), (32 * i + 12, 32 * j + 12))
                if state[i][j][4] == 1:
                    draw.text((32 * i + 13, 32 * j + 11), 'X', fill=255)
                if state[i][j][5] == 1:
                    draw.text((32 * i + 13, 32 * j + 11), 'O', fill=255)
        return image


def expand_action(action):
    return [action[0], ACTIONS[action[1]][0], ACTIONS[action[1]][1], ACTIONS[action[2]][0], ACTIONS[action[2]][1]]
