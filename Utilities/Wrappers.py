import ast

import gym

from Utilities.Santorini import ACTIONS


class OpponentWrapper(gym.Wrapper):
    def __init__(self, env, agent_type):
        super().__init__(env)
        self.agent_type = agent_type
        assert agent_type in ['Random', 'MinMax', 'MinMaxRandom', 'MonteCarlo'], 'Select a valid opponent'

    def step(self, action, selfplay=False):
        action = to_action(action, self.env.name)
        obs, reward, done, info = self.env.step(action)
        render = self.env.render_board(self.env.get_fixed_obs())
        if done:
            if selfplay:
                return obs, reward, done, {'info': info}, render, None
            return obs, reward, done, {'info': info}, render
        action_opponent = self.get_opponent_action(action)
        obs_adv, reward_adv, done_adv, info_adv = self.env.step(action_opponent)
        if selfplay:
            return obs_adv, reward_adv, done_adv, {'info': info_adv}, render, action_opponent
        return obs_adv, reward_adv, done_adv, {'info': info_adv}, render

    def get_opponent_action(self, action=None):
        if self.agent_type == 'Random':
            if self.env.name == 'Santorini':
                return self.env.get_random_action(self.env.state, self.env.player_one_workers,
                                                  self.env.player_two_workers, self.env.player_one)
            else:
                return self.env.get_random_action(self.env.state)
        elif self.agent_type == 'MinMax':
            return self.env.minmax(self.env.state)
        elif self.agent_type == 'MinMaxRandom':
            return self.env.minmaxran(self.env.state)
        elif self.agent_type == 'MonteCarlo':
            node = self.env.mc_node
            if action is not None:
                if len(node.children) == 0:
                    node.expand(self.env)
                node = self.env.mc_node.children[str(action)]
            for _ in range(self.env.normal_simulations):
                node.rollout_simulation(node.state, node.player_one_workers, node.player_two_workers,
                                        node.player_one, self.env, 4)
            action, node = node.best_move(self.env.agent_first)
            self.env.mc_node = node
            return ast.literal_eval(action)
        else:
            raise ValueError('Opponent provided does not exist!')

    def reset(self, selfplay=False, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.env.agent_first:
            opponent_action = self.get_opponent_action()
            obs, _, _, _ = self.env.step(opponent_action)
            if selfplay:
                return obs, opponent_action
        return obs


def to_action(value, game):
    if game == "TicTacToe":
        return value
    elif game == "ConnectFour":
        return value
    elif game == "Santorini":
        if type(value) == list:
            return value
        worker = value // 64
        movement = ACTIONS[(value % 64) // 8]
        build = ACTIONS[(value % 64) % 8]
        return [worker, movement[0], movement[1], build[0], build[1]]
