import gym


class Agent:
    def __init__(self, observation_space, action_space, batch_size, checkpoint_dir='tmp/agent_name'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.state_space_shape = self.observation_space.shape
        if type(self.action_space) == gym.spaces.Discrete:
            self.action_space_shape = (self.action_space.n,)
            self.action_number = 1
        else:
            self.action_space_shape = self.action_space.nvec
            self.action_number = self.action_space.shape[0]
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.checkpoint_dir = checkpoint_dir

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
