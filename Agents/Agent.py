import gym


class Agent:
    def __init__(self, environment, stacking_dict, batch_size, checkpoint_dir='tmp/agent_name', seed=42):
        self.environment = environment
        self.stacking_dict = stacking_dict
        if self.stacking_dict is not None:
            self.state_space_shape = self.stacking_dict['state_space_shape']
        else:
            self.state_space_shape = self.environment.observation_space.shape
        if type(self.environment.action_space) == gym.spaces.Discrete:
            self.action_space_shape = (self.environment.action_space.n,)
            self.action_number = 1
        else:
            self.action_space_shape = self.environment.action_space.nvec
            self.action_number = self.environment.action_space.shape[0]
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

    def act(self, observation):
        raise NotImplementedError("act method needs to be implemented by subclasses")

    def store(self, initial_state, action, reward, final_state, terminal):
        raise NotImplementedError("store method needs to be implemented by subclasses")

    def learn(self):
        raise NotImplementedError("learn method needs to be implemented by subclasses")

    def save(self):
        raise NotImplementedError("save method needs to be implemented by subclasses")

    def load(self):
        raise NotImplementedError("load method needs to be implemented by subclasses")
