import itertools

import gym
import numpy as np
from PIL import Image, ImageDraw
from gym import spaces
from tqdm import tqdm

from Utilities.MCTS import MC_Tree

BOARD_SIZE = 5

ACTION_SPACE = 128  # np.array([2, 3, 3, 3, 3, 5, 5])
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
WIDTH = 160
HEIGHT = 160


class SantoriniEnv(gym.Env):
    def __init__(self, representation, agent_first, random_init=True, mcts=False, initial_simulations=0,
                 normal_simulations=0):
        self.name = "Santorini"
        self.possible_actions = list(itertools.product(*[[0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        self.representation = representation
        self.action_space = spaces.Discrete(ACTION_SPACE)
        if self.representation == 'Tabular':
            self.observation_space = spaces.Box(low=-1, high=1, shape=(BOARD_SIZE, BOARD_SIZE, len(LAYERS), 1),
                                                # LOW -1
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0., high=1., shape=(HEIGHT, WIDTH, 1), dtype=np.float32)
        self.player_one = True
        self.turn = 0
        self.random_init = random_init
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, len(LAYERS))).astype("int8")
        self.player_one_workers = [[], []]
        self.player_two_workers = [[], []]
        self.agent_first = agent_first
        assert self.representation in ['Tabular', 'Graphic']
        self.done = False
        self.exploration_parameter = 10  # math.sqrt(2)
        if self.random_init:
            self.turn = 4
            self._assign_worker(None)
        self.mcts = mcts
        self.initial_simulations = initial_simulations
        self.normal_simulations = normal_simulations
        self.mc_node = MC_Tree(None, self.state, self.player_one_workers, self.player_two_workers, self.player_one)
        if mcts:
            for _ in tqdm(range(self.initial_simulations)):
                self.mc_node.rollout_simulation(self.state, self.player_one_workers, self.player_two_workers, self.player_one, self)
        self.exploration_parameter = 1

    def reset(self):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE, len(LAYERS))).astype("int8")
        self.player_one = True
        self.turn = 0
        self.done = False

        if self.random_init:
            self.turn = 4
            self._assign_worker(None)
        while not self.mc_node.is_root():
            self.mc_node = self.mc_node.parent
        return self._get_observation()

    def _get_observation(self):
        obs = self.get_fixed_obs()
        if self.representation == 'Tabular':
            return obs
        else:
            return np.expand_dims(np.asarray(self.render_board(obs)), axis=-1) / 255.

    def get_fixed_obs(self):
        obs = np.expand_dims(self.state, axis=-1)
        if not self.agent_first:
            obs = np.concatenate([obs[:, :, 0:4, :], obs[:, :, 5:6, :], obs[:, :, 4:5, :]],
                                 axis=-2)  # FIX player on last 2 layers
        return obs

    def _assign_worker(self, player_num, action=None):
        if self.random_init:
            self.player_one_workers[0] = [1, 1]
            self.state[1][1][LAYERS['player1']] = 1
            self.player_one_workers[1] = [3, 3]
            self.state[3][3][LAYERS['player1']] = -1
            self.player_two_workers[0] = [1, 3]
            self.state[1][3][LAYERS['player2']] = 1
            self.player_two_workers[1] = [3, 1]
            self.state[3][1][LAYERS['player2']] = -1
            return
        elif action is not None:
            if all(self.state[action[5]][action[6]][LAYERS['player1']:] == 0):
                self.state[action[5]][action[6]][LAYERS['player1'] + player_num] = 1
                return [action[5], action[6]], False
            else:
                return [], True
        return

    # OpenAI Gym Environments standard function which returns next state given the action to perform,
    # as well as the state of the game (Terminal/non-Terminal), action reward and additional information
    def step(self, action):

        if action is not None and len(action) == 3:  # 2,8,8 ---> 2, 3,3, 3,3
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

        if not self.check_valid_action(self.state, action, self.player_one_workers, self.player_two_workers,
                                       self.player_one):
            reward = -2 * ONE_REWARD if player_num == 0 else -2 * TWO_REWARD
            info = "Illegal action by player 1" if player_num == 0 else "Illegal action by player 2"
            done = True
            if not self.agent_first:
                reward = -reward
            return self._get_observation(), reward, done, info

        self.state, self.player_one_workers, self.player_two_workers, self.player_one, self.turn = self.result(
            self.state, action, self.player_one_workers, self.player_two_workers, self.player_one, self.turn)

        reward, done, info = self.goal(self.state, self.player_one_workers, self.player_two_workers, self.player_one)
        if not self.agent_first:
            reward = -reward
        return self._get_observation(), reward, done, info

    def result(self, state, action, player_one_workers, player_two_workers, player_one, turn):

        state_copy = state.copy()
        player_one_workers_copy = player_one_workers.copy()
        player_two_workers_copy = player_two_workers.copy()

        player_num = 0 if player_one else 1

        coord_worker = player_one_workers_copy[action[0]] if player_one else player_two_workers_copy[action[0]]

        state_copy[coord_worker[0]][coord_worker[1]][LAYERS['player1'] + player_num] = 0

        state_copy[coord_worker[0] + action[1]][coord_worker[1] + action[2]][LAYERS['player1'] + player_num] = 1 if \
        action[0] == 0 else -1

        if player_one:
            player_one_workers_copy[action[0]] = [coord_worker[0] + action[1], coord_worker[1] + action[2]]
        else:
            player_two_workers_copy[action[0]] = [coord_worker[0] + action[1], coord_worker[1] + action[2]]

        coord_worker = player_one_workers_copy[action[0]] if player_one else player_two_workers_copy[action[0]]

        state_copy = self._build(state_copy, coord_worker[0] + action[3], coord_worker[1] + action[4])

        player_one = not player_one
        turn += 1
        return state_copy, player_one_workers_copy, player_two_workers_copy, player_one, turn

    def _build(self, state, build_row, build_column):
        state_copy = state.copy()
        level = 0
        while state_copy[build_row][build_column][level] == 1 and level < 3:
            level += 1
        state_copy[build_row][build_column][level] = 1
        return state_copy

    # Returns True if the action is valid, else False
    def check_valid_action(self, state, action, player_one_workers, player_two_workers, player_one):
        state_copy = state.copy()
        player_one_workers_copy = player_one_workers.copy()
        player_two_workers_copy = player_two_workers.copy()

        if len(action) == 3:  # 2,8,8 ---> 2, 3,3, 3,3
            action = expand_action(action)

        if action[1] == action[2] == 0 or action[3] == action[4] == 0:
            return False  # invalid action no movement/build on spot

        coord_worker = player_one_workers_copy[action[0]] if player_one else player_two_workers_copy[action[0]]
        coord_landing = [coord_worker[0] + action[1], coord_worker[1] + action[2]]
        coord_building = [coord_landing[0] + action[3], coord_landing[1] + action[4]]

        if any(i > 4 for i in coord_landing) or any(i < 0 for i in coord_landing):
            return False  # move out of list

        if any(i > 4 for i in coord_building) or any(i < 0 for i in coord_building):
            return False  # building out of list

        if any(state_copy[coord_landing[0]][coord_landing[1]][3:]) != 0:
            return False  # no move for occupied or dome

        if any(state_copy[coord_building[0]][coord_building[1]][
               3:]) != 0 and coord_building != coord_worker:  # Second term allows to build where the worker was before moving
            return False  # no building for occupied or dome

        if sum(state_copy[coord_landing[0]][coord_landing[1]][:3]) > sum(
                state_copy[coord_worker[0]][coord_worker[1]][:3]) + 1:
            return False  # too high destination

        return True

    # Returns all possible actions given the state in the expanded form
    def actions(self, state, player_one_workers, player_two_workers, player_one):
        actions = []
        for action in itertools.product(*[[0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]):
            action = list(action)
            if self.check_valid_action(state, action, player_one_workers, player_two_workers, player_one):
                actions.append(action)
        return actions

    def get_random_action(self, state, player_one_workers, player_two_workers, player_one):
        np.random.shuffle(self.possible_actions)
        for a in self.possible_actions:
            if self.check_valid_action(state, a, player_one_workers, player_two_workers, player_one):
                return a
        return None

    # Checks whether a final state is reached
    def goal(self, state, player_one_workers, player_two_workers, player_one):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i][j][LAYERS['third']] == 1:
                    if state[i][j][LAYERS['player1']] != 0:
                        return ONE_REWARD, True, "player one wins"
                    elif state[i][j][LAYERS['player2']] != 0:
                        return TWO_REWARD, True, "player two wins"
        if len(self.actions(state, player_one_workers, player_two_workers, True)) == 0:
            return TWO_REWARD, True, "player 1 doesn't have legal moves, player 2 wins"
        elif len(self.actions(state, player_one_workers, player_two_workers, False)) == 0:
            return ONE_REWARD, True, "player 2 doesn't have legal moves, player 1 wins"
        return 0, False, "game not end"

    def render_board(self, state):
        image = Image.new('L', (WIDTH, HEIGHT))
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
                if state[i][j][4] != 0:
                    if state[i][j][4] == 1:
                        draw.text((32 * i + 13, 32 * j + 11), 'A', fill=255)
                    elif state[i][j][4] == -1:
                        draw.text((32 * i + 13, 32 * j + 11), 'B', fill=255)
                if state[i][j][5] != 0:
                    if state[i][j][5] == 1:
                        draw.text((32 * i + 13, 32 * j + 11), 'C', fill=255)
                    elif state[i][j][5] == -1:
                        draw.text((32 * i + 13, 32 * j + 11), 'D', fill=255)
        return image


def expand_action(action):
    return [action[0], ACTIONS[action[1]][0], ACTIONS[action[1]][1], ACTIONS[action[2]][0], ACTIONS[action[2]][1]]
