import copy
import math
import numpy as np
import tensorflow as tf
import random

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


class MC_Tree:
    def __init__(self, parent, state=None, player_one_workers=None, player_two_workers=None, player_one=None):
        self.state = state
        self.player_one_workers = player_one_workers
        self.player_two_workers = player_two_workers
        self.player_one = player_one

        self.parent = parent
        self.children = {}  # one node for each action map: str(action):node
        self.W = 0  # Wins
        self.N = 0  # Simulations
        self.Q = 1  # Wins/Simulations + exploration_parameter*sqrt(log(parent.N)/N)

    def expand(self, environment):
        if environment.name == "TicTacToe" or environment.name == "ConnectFour":
            actions = environment.actions(self.state)
            for action in actions:
                if str(action) not in self.children:
                    state = environment.result(self.state, action)
                    self.children[str(action)] = MC_Tree(state=state.copy(), parent=self)

        # if environment.name == "ConnectFour":
        #    actions = environment.actions(self.state)
        #    for action in actions:
        #        if str(action) not in self.children:
        #            state = environment.result(self.state, action)
        #            self.children[str(action)] = MC_Tree(state=state.copy(), parent=self)

        if environment.name == "Santorini":
            actions = environment.actions(self.state, self.player_one_workers, self.player_two_workers, self.player_one)
            for action in actions:
                if str(action) not in self.children:
                    state, player_one_workers, player_two_workers, player_one, _ = environment.result(self.state,
                                                                                                      action,
                                                                                                      self.player_one_workers,
                                                                                                      self.player_two_workers,
                                                                                                      self.player_one,
                                                                                                      4)
                    self.children[str(action)] = MC_Tree(state=state.copy(),
                                                         player_one_workers=player_one_workers.copy(),
                                                         player_two_workers=player_two_workers.copy(),
                                                         player_one=player_one, parent=self)

    # returns action and next node
    def select_next(self):
        return max(self.children.items(), key=lambda node: node[1].Q)

    # Select action and next node based only on exploitation (no exploration factor)
    def best_move(self):
        return max(self.children.items(),
                   key=lambda node: node[1].W / node[1].N if node[1].N else 0)  # if number of simulations = 0 return 0

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def rollout_simulation(self, state=None, player_one_workers=None, player_two_workers=None, player_one=None,
                           environment=None, turn=4):  # turn=4 prevents to simulate initialization (fix?)

        # Exploration
        to_update = []
        node = self
        while not node.is_root():
            node = node.parent
            to_update.append(node)

        to_update.reverse()
        node = self
        to_update.append(node)

        while not node.is_leaf():
            _, node = node.select_next()
            to_update.append(node)

            # Expansion
        if (not environment.goal(node.state, node.player_one_workers, node.player_two_workers, node.player_one)[1]) if \
                environment.name == "Santorini" else (not environment.goal(node.state)[1]):
            node.expand(environment)

            # Simulation
        state = node.state.copy()
        player_one_workers = node.player_one_workers.copy()
        player_two_workers = node.player_two_workers.copy()
        player_one = node.player_one

        if environment.name == "TicTacToe":
            while not environment.goal(state)[1]:
                actions = environment.actions(state)
                if len(actions) > 0:
                    state = environment.result(state, actions[np.random.choice(len(actions))])
                else:
                    break
            player_one = (1 if np.count_nonzero(state == 1) == np.count_nonzero(state == -1) else -1) == 1
            reward = player_one == environment.agent_first

        if environment.name == "ConnectFour":
            while not environment.goal(state)[1]:
                actions = environment.actions(state)
                if len(actions) > 0:
                    state = environment.result(state)
                else:
                    break
            player_one = (1 if np.count_nonzero(state == 1) == np.count_nonzero(state == -1) else -1) == 1
            reward = player_one == environment.agent_first

        if environment.name == "Santorini":
            while not environment.goal(state, player_one_workers, player_two_workers, player_one)[1]:
                actions = environment.actions(state, player_one_workers, player_two_workers, player_one)
                if len(actions) > 0:
                    state, player_one_workers, player_two_workers, player_one, _ = environment.result(state, actions[
                        np.random.choice(len(actions))], player_one_workers, player_two_workers, player_one, 4)
                else:
                    break
            reward = player_one == environment.agent_first  # not player_one # True se vince 1 , False se vince 2 # Player_one è chi perde, siamo contenti se player_one è diverso
        # reward = environment.goal(state)[0] # 1 if player one wins else -1

        # Backpropagation
        for node in to_update:
            node.update(reward, environment.exploration_parameter)

    def update(self, reward, exploration_parameter):
        self.N += 1
        if reward:
            self.W += 1
        else:
            self.W -= 1

        if not self.is_root():
            self.Q = self.W / self.N + exploration_parameter * math.sqrt(math.log(self.parent.N) / self.N)

    def get_relative_state(self, env_name, player):
        if env_name == "TicTacToe" or env_name == "ConnectFour":
            if player == 0:
                return self.state
            else:
                return -self.state

        elif env_name == "Santorini":
            if self.player_one:
                return self.state
            else:
                state = np.expand_dims(self.state, axis=-1)
                state = np.concatenate([state[:, :, :4, :], state[:, :, 5:6, :], state[:, :, 4:5, :]], axis=-2)
                return state.squeeze()


class SelfPlayMCTS(MC_Tree):
    def __init__(self, prior_action, value, parent, Cpuct=1, state=None, player_one_workers=None,
                 player_two_workers=None,
                 player_one=None):
        super(SelfPlayMCTS, self).__init__(parent, state, player_one_workers, player_two_workers, player_one)
        self.Cpuct = Cpuct
        self.prior_actions = None  # list of prior on next actions given by the nnet
        self.P = prior_action
        self.V = value
        self.Q = 0

    def rollout_simulation(self, environment, network):
        to_update = [self]
        node = self
        while not node.is_leaf():
            _, node = node.select_next()
            to_update.append(node)

        # Expansion
        if environment.name == "Santorini":
            done = environment.goal(node.state, node.player_one_workers, node.player_two_workers, node.player_one)[1]
        else:
            done = environment.goal(node.state)[1]

        if not done:
            node.expand(environment)
            node.evaluate(network, environment)
        else:
            value = environment.goal(node.state, node.player_one_workers, node.player_two_workers, node.player_one)[0] \
                if environment.name == "Santorini" else environment.goal(node.state)[0]
            # -1 se vince il secondo, +1 se vince il primo
            # player_one == True se primo False se secondo
            node.V = value # -> player_one 1, player_two -1
            # se la partita è finita, non è il turno di chi ha vinto -> -environment.goal(node.state)[0]

        # Backpropagation
        to_update.reverse()
        value = node.V
        for n in to_update:
            n.update(value)
            value = -value

    def evaluate(self, network, env):
        if env.representation == "Graphic":
            predictions = network.predict(tf.expand_dims(tf.convert_to_tensor(env.render_board(self.get_relative_state(env.name, self.player_one))), axis = 0))
        else:
            predictions = network.predict(x=state_to_input_model(self.get_relative_state(env.name, self.player_one), env), verbose=0)  # reshape for the network
        value = predictions[0]
        prior_actions = predictions[1].squeeze()
        if self.is_root():
            self.prior_actions = 0.75 * prior_actions + 0.25 * np.random.dirichlet([0.03] * len(prior_actions))
        self.prior_actions = prior_actions
        self.V = value

        for i in range(len(prior_actions)):  # len(prior_actions) == # actions of the env
            action = to_action(i, env.name)
            if str(action) in self.children:
                self.children[str(action)].P = prior_actions[i]

    def expand(self, environment):
        if environment.name == "TicTacToe" or environment.name == "ConnectFour":
            actions = environment.actions(self.state)
            for action in actions:
                if str(action) not in self.children:
                    state = environment.result(self.state, action)
                    self.children[str(action)] = SelfPlayMCTS(prior_action=None,
                                                              value=None,
                                                              state=state.copy(),
                                                              parent=self)

        if environment.name == "Santorini":
            actions = environment.actions(self.state, self.player_one_workers, self.player_two_workers, self.player_one)
            for action in actions:
                if str(action) not in self.children:
                    state, player_one_workers, player_two_workers, player_one, _ = environment.result(self.state,
                                                                                                      action,
                                                                                                      self.player_one_workers,
                                                                                                      self.player_two_workers,
                                                                                                      self.player_one,
                                                                                                      4)
                    self.children[str(action)] = SelfPlayMCTS(prior_action=None,
                                                              value=None,
                                                              state=state.copy(),
                                                              player_one_workers=player_one_workers.copy(),
                                                              player_two_workers=player_two_workers.copy(),
                                                              player_one=player_one,
                                                              parent=self)

    def select_next(self):
        return max(self.children.items(), key=lambda node: node[1].Q + self.Cpuct * node[1].P * math.sqrt(self.N) / (
                1 + node[1].N))
        # at = argmax(Q(st,a) + U(st,a)  where U(st, a) = Cpuct * P(s,a) * (sum(N(s,b))^0.5) / (1 + N(s,a))

    def update(self, value):
        #v = value if #self.player_one == winner else -value
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


def to_action(value, game):
    if game == "TicTacToe":
        return value
    elif game == "ConnectFour":
        return value
    elif game == "Santorini":
        worker = value // 64
        movement = ACTIONS[(value % 64) // 8]
        build = ACTIONS[(value % 64) % 8]
        return [worker, movement[0], movement[1], build[0], build[1]]


def state_to_input_model(state, env):
    if env.name == "TicTacToe":
        return state.reshape(-1, 3, 3, 1) if env.representation == "Tabular" else state
    elif env.name == "ConnectFour":
        return state.reshape(-1, 6, 7, 1) if env.representation == "Tabular" else state
    elif env.name == "Santorini":
        return state.reshape(-1, 5, 5, 6) if env.representation == "Tabular" else state#.reshape(-1, 160, 160, 1)
