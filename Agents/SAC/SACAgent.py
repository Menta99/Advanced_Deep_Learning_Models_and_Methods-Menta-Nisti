import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from Agents.Agent import Agent
from Agents.SAC.Network import SingleNetwork
from Utilities.NetworkBuilder import NetworkBuilder
from Utilities.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer


def update_network_parameters(network, target_network, target_update_coefficient):
    weights = [w * target_update_coefficient + target_network.weights[i] * (1 - target_update_coefficient)
               for i, w in enumerate(network.weights)]
    target_network.set_weights(weights)


class SACAgent(Agent):
    def __init__(self, observation_space, action_space, actor_net_dict, critic_net_dict,
                 net_update=1, discount_factor=0.99, actor_net_optimizer=tf.keras.optimizers.Adam,
                 critic_net_optimizer=tf.keras.optimizers.Adam, actor_net_learning_rate=1e-4,
                 critic_net_learning_rate=1e-4, actor_net_loss=tf.keras.losses.Huber(),
                 critic_net_loss=tf.keras.losses.Huber(), num_episodes=100000, learning_starts=1000,
                 memory_size=32768, memory_alpha=0, memory_beta=0, max_epsilon=1.0, min_epsilon=0.05,
                 epsilon_a=0.06, epsilon_b=0.05, epsilon_c=1.5, batch_size=64, max_norm_grad=5, tau=0.005,
                 entropy_coeff=None, initial_entropy_coeff=50., checkpoint_dir=''):
        super(SACAgent, self).__init__(observation_space, action_space, batch_size, checkpoint_dir)
        self.actor_net_dict = actor_net_dict
        self.critic_net_dict = critic_net_dict
        self.net_update = net_update
        self.discount_factor = discount_factor
        self.actor_net_optimizer = actor_net_optimizer
        self.critic_net_optimizer = critic_net_optimizer
        self.actor_net_learning_rate = actor_net_learning_rate
        self.critic_net_learning_rate = critic_net_learning_rate
        self.actor_net_loss = actor_net_loss
        self.critic_net_loss = critic_net_loss
        self.num_episodes = num_episodes
        self.learning_starts = learning_starts
        self.memory_size = memory_size
        self.memory_alpha = memory_alpha
        if memory_alpha == 0:
            self.memory_beta = 0
            self.memory = ReplayBuffer(self.memory_size, self.observation_space, self.action_space)
        else:
            self.memory_beta = memory_beta
            self.memory = PrioritizedReplayBuffer(self.memory_size, self.observation_space,
                                                  self.action_space, self.memory_alpha)
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_a = epsilon_a
        self.epsilon_b = epsilon_b
        self.epsilon_c = epsilon_c
        self.max_norm_grad = max_norm_grad
        self.tau = tau
        if entropy_coeff is None:
            self.automatic_entropy_tuning = True
            self.target_entropy = -np.log((1.0 / self.action_number)) * 0.98
            self.alpha = tf.Variable(initial_entropy_coeff, dtype=np.float32)
            self.log_alpha = tf.Variable(0., dtype=np.float32)
            self.log_alpha.assign(tf.math.log(self.alpha))
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_net_learning_rate, epsilon=1e-4)
        else:
            self.automatic_entropy_tuning = False
            self.alpha = entropy_coeff

        self.network_builder = NetworkBuilder()
        self.actor_net = self._init_network(self.actor_net_dict, 'actor_net', self.checkpoint_dir,
                                            self.actor_net_optimizer, self.actor_net_learning_rate,
                                            self.actor_net_loss)
        self.local_critic_1_net = self._init_network(self.critic_net_dict, 'local_critic_1_net', self.checkpoint_dir,
                                                     self.critic_net_optimizer, self.critic_net_learning_rate,
                                                     self.critic_net_loss)
        self.local_critic_2_net = self._init_network(self.critic_net_dict, 'local_critic_2_net', self.checkpoint_dir,
                                                     self.critic_net_optimizer, self.critic_net_learning_rate,
                                                     self.critic_net_loss)
        self.target_critic_1_net = self._init_network(self.critic_net_dict, 'target_critic_1_net', self.checkpoint_dir,
                                                      self.critic_net_optimizer, self.critic_net_learning_rate,
                                                      self.critic_net_loss)
        self.target_critic_2_net = self._init_network(self.critic_net_dict, 'target_critic_2_net', self.checkpoint_dir,
                                                      self.critic_net_optimizer, self.critic_net_learning_rate,
                                                      self.critic_net_loss)
        self.update_target(1)
        self.last_loss = np.inf

    def _init_network(self, network_dict, name, checkpoint_dir, optimizer, learning_rate, loss):
        network = SingleNetwork(self.network_builder.build_network(network_dict), name, checkpoint_dir)
        network.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
        return network

    def update_target(self, coefficient):
        update_network_parameters(self.local_critic_1_net, self.target_critic_1_net, coefficient)
        update_network_parameters(self.local_critic_2_net, self.target_critic_2_net, coefficient)

    def store(self, initial_state, action, reward, final_state, terminal):
        self.memory.push(initial_state, action, reward, final_state, terminal)

    def epsilon_scheduler(self):
        standardized_time = (self.time_step - self.epsilon_a * self.num_episodes) / (self.epsilon_c * self.num_episodes)
        cosh = np.cosh(np.math.exp(-standardized_time))
        epsilon = 1.1 - (1 / cosh + (self.time_step * self.epsilon_c / self.num_episodes))
        return max(min(epsilon, self.max_epsilon), self.min_epsilon)

    def act(self, observation, test_mode=False):
        if np.random.random() < self.epsilon_scheduler() and not test_mode:
            actions = [np.random.choice(self.action_space_shape[i]) for i in range(self.action_number)]
        else:
            state = tf.expand_dims(tf.convert_to_tensor(observation), axis=0)
            actions = [np.argmax(tensor, axis=-1) for tensor in self.actor_net(state)]
        if not test_mode:
            self.time_step += 1
        return actions

    def generate_action_sample(self, state):
        action_probabilities = self.actor_net(state)
        max_probability_action = tf.argmax(action_probabilities, axis=-1)
        action_distribution = tfp.distributions.Categorical(action_probabilities)
        action = action_distribution.sample()
        z = tf.cast(action_probabilities == 0.0, dtype=np.float32) * 1e-8
        log_action_probabilities = tf.math.log(action_probabilities + z)
        return action, action_probabilities, log_action_probabilities, max_probability_action

    def learn(self):
        if self.memory.counter < self.batch_size and not self.memory.full:
            return

        if self.time_step % self.net_update == 0 and self.time_step >= self.learning_starts:
            if isinstance(self.memory, PrioritizedReplayBuffer):
                initial_states, actions, rewards, final_states, \
                terminals, weights, indexes = self.memory.pop(self.batch_size, self.memory_beta)
            else:
                initial_states, actions, rewards, final_states, \
                terminals, weights, indexes = self.memory.pop(self.batch_size)

            _, action_probabilities, log_action_probabilities, _ = self.generate_action_sample(final_states)
            q_target_1_value = self.target_critic_1_net(final_states)
            q_target_2_value = self.target_critic_2_net(final_states)
            min_q_target = tf.minimum(q_target_1_value, q_target_2_value)
            target_update = tf.reduce_sum(action_probabilities *
                                          (min_q_target - self.alpha * log_action_probabilities), axis=1)
            target_q_values = tf.expand_dims(rewards + self.discount_factor * target_update * (1 - terminals), axis=-1)

            with tf.GradientTape(persistent=True) as tape:
                current_q1_value = tf.gather(self.local_critic_1_net(initial_states), tf.cast(actions, dtype=tf.int32),
                                             batch_dims=1)
                current_q2_value = tf.gather(self.local_critic_2_net(initial_states), tf.cast(actions, dtype=tf.int32),
                                             batch_dims=1)
                q1_loss = self.local_critic_1_net.loss(target_q_values, current_q1_value, sample_weight=weights)
                q2_loss = self.local_critic_2_net.loss(target_q_values, current_q2_value, sample_weight=weights)

            self._update_network(self.local_critic_1_net, tape, q1_loss)
            self._update_network(self.local_critic_2_net, tape, q2_loss)
            update_network_parameters(self.local_critic_1_net, self.target_critic_1_net, self.tau)
            update_network_parameters(self.local_critic_2_net, self.target_critic_2_net, self.tau)

            with tf.GradientTape() as tape2:
                action, action_probabilities, log_action_probabilities, _ = self.generate_action_sample(initial_states)
                current_q1_value_actor = self.local_critic_1_net(initial_states)
                current_q2_value_actor = self.local_critic_2_net(initial_states)
                min_current_q_value_actor = tf.minimum(current_q1_value_actor, current_q2_value_actor)
                inside_term = self.alpha * log_action_probabilities - min_current_q_value_actor
                actor_loss = tf.reduce_mean(tf.reduce_sum(action_probabilities * inside_term, axis=-1))
                log_action_probabilities = tf.reduce_sum(log_action_probabilities * action_probabilities)
                self.last_loss = actor_loss

            self._update_network(self.actor_net, tape2, actor_loss)
            if self.automatic_entropy_tuning:
                with tf.GradientTape() as tape3:
                    entropy_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_action_probabilities +
                                                                                     self.target_entropy))
                entropy_gradients = tape3.gradient(entropy_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(entropy_gradients, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

            if isinstance(self.memory, PrioritizedReplayBuffer):
                self.memory.update_priorities(indexes, tf.math.abs(target_q_values -
                                                                   tf.minimum(current_q1_value, current_q2_value)))
            self.learn_step_counter += 1

    def _update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        if self.max_norm_grad is not None:
            network_gradient, _ = tf.clip_by_global_norm(network_gradient, self.max_norm_grad)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def save(self):
        print('Saving models and parameters...')
        #f = open(os.path.join(self.checkpoint_dir, '_params'), "wb")
        #pickle.dump([self.memory, self.learn_step_counter, self.time_step], f)
        #f.close()
        self.actor_net.save_weights(self.actor_net.checkpoint_file)
        self.local_critic_1_net.save_weights(self.local_critic_1_net.checkpoint_file)
        self.local_critic_2_net.save_weights(self.local_critic_2_net.checkpoint_file)
        self.target_critic_1_net.save_weights(self.target_critic_1_net.checkpoint_file)
        self.target_critic_2_net.save_weights(self.target_critic_2_net.checkpoint_file)

    def load(self):
        print('Loading models and parameters...')
        #f = open(os.path.join(self.checkpoint_dir, '_params'), "rb")
        #self.memory, self.learn_step_counter, self.time_step = pickle.load(f)
        #f.close()
        self.actor_net.load_weights(self.actor_net.checkpoint_file)
        self.local_critic_1_net.load_weights(self.local_critic_1_net.checkpoint_file)
        self.local_critic_2_net.load_weights(self.local_critic_2_net.checkpoint_file)
        self.target_critic_1_net.load_weights(self.target_critic_1_net.checkpoint_file)
        self.target_critic_2_net.load_weights(self.target_critic_2_net.checkpoint_file)
