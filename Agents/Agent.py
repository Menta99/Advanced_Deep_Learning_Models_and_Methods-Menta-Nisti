class Agent:
    def __init__(self, environment, batch_size, checkpoint_dir='tmp/agent_name', seed=42):
        self.environment = environment
        self.state_space_shape = self.environment.observation_space.shape
        self.action_space_shape = self.environment.action_space.shape
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
