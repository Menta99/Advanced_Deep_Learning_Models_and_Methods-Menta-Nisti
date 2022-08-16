import os
import pickle

from Agents.Agent import Agent
from Agents.DDDQN.Network import DQNetwork, DuelingNetwork
from Utilities.NetworkBuilder import NetworkBuilder
from Utilities.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
import tensorflow as tf
import numpy as np


def update_network_parameters(network, target_network, target_update_coefficient):
    weights = [w * target_update_coefficient + target_network.weights[i] * (1 - target_update_coefficient)
               for i, w in enumerate(network.weights)]
    target_network.set_weights(weights)


class DDDQNAgent(Agent):
    def __init__(self, observation_space, action_space, q_net_dict, q_target_net_dict, double_q=False, dueling_q=False,
                 q_net_update=4, q_target_net_update=10000, discount_factor=0.99,
                 q_net_optimizer=tf.keras.optimizers.Adam, q_target_net_optimizer=tf.keras.optimizers.Adam,
                 q_net_learning_rate=1e-4, q_target_net_learning_rate=1e-4, q_net_loss=tf.keras.losses.Huber(),
                 q_target_net_loss=tf.keras.losses.Huber(), num_episodes=100000, learning_starts=1000,
                 memory_size=32768, memory_alpha=0, memory_beta=0, max_epsilon=1.0, min_epsilon=0.05, epsilon_a=0.06,
                 epsilon_b=0.05, epsilon_c=1.5, batch_size=32, max_norm_grad=10, tau=1, checkpoint_dir=''):
        super(DDDQNAgent, self).__init__(observation_space, action_space, batch_size, checkpoint_dir)
        self.q_net_dict = q_net_dict
        self.q_target_net_dict = q_target_net_dict
        self.double_q = double_q
        self.dueling_q = dueling_q
        self.q_net_update = q_net_update
        self.q_target_net_update = q_target_net_update
        self.discount_factor = discount_factor
        self.q_net_optimizer = q_net_optimizer
        self.q_target_net_optimizer = q_target_net_optimizer
        self.q_net_learning_rate = q_net_learning_rate
        self.q_target_net_learning_rate = q_target_net_learning_rate
        self.q_net_loss = q_net_loss
        self.q_target_net_loss = q_target_net_loss
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

        self.network_builder = NetworkBuilder()
        self.q_net = self._init_network(self.q_net_dict, 'q_net', self.checkpoint_dir, self.q_net_optimizer,
                                        self.q_net_learning_rate, self.q_net_loss)

        self.q_target_net = self._init_network(self.q_target_net_dict, 'q_target_net', self.checkpoint_dir,
                                               self.q_target_net_optimizer, self.q_target_net_learning_rate,
                                               self.q_target_net_loss)

        update_network_parameters(self.q_net, self.q_target_net, self.tau)
        self.q_target_net.trainable = False
        self.last_loss = np.inf

    def _init_network(self, network_dict, name, checkpoint_dir, optimizer, learning_rate, loss):
        if self.dueling_q:
            base, adv, val = network_dict
            network = DuelingNetwork(self.network_builder.build_network(base),
                                     self.network_builder.build_network(adv),
                                     self.network_builder.build_network(val), name, checkpoint_dir)
        else:
            network = DQNetwork(self.network_builder.build_network(network_dict), name, checkpoint_dir)
        network.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
        return network

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
            if self.dueling_q:
                actions = [np.argmax(tensor, axis=-1) for tensor in self.q_net.advantage(state)]
            else:
                actions = [np.argmax(tensor, axis=-1) for tensor in self.q_net(state)]

        if not test_mode:
            self.time_step += 1
        return actions

    def learn(self):
        if self.memory.counter < self.batch_size and not self.memory.full:
            return

        if self.time_step % self.q_target_net_update == 0:
            update_network_parameters(self.q_net, self.q_target_net, self.tau)

        if self.time_step % self.q_net_update == 0 and self.time_step >= self.learning_starts:
            if isinstance(self.memory, PrioritizedReplayBuffer):
                initial_states, actions, rewards, final_states, \
                    terminals, weights, indexes = self.memory.pop(self.batch_size, self.memory_beta)
            else:
                initial_states, actions, rewards, final_states, \
                    terminals, weights, indexes = self.memory.pop(self.batch_size)

            if self.double_q:
                q_next = self.q_target_net(final_states)
            else:
                q_next = self.q_net(final_states)
            max_actions = tf.math.argmax(self.q_net(final_states), axis=1)
            target_q_values = tf.expand_dims(rewards + self.discount_factor *
                                             tf.gather(q_next, max_actions, batch_dims=1) * (1 - terminals), axis=-1)

            with tf.GradientTape() as tape:
                current_q_values = tf.gather(self.q_net(initial_states), tf.cast(actions, dtype=tf.int32), batch_dims=1)
                loss = self.q_net_loss(target_q_values, current_q_values, sample_weight=weights)
                self.last_loss = loss

            self._update_network(self.q_net, tape, loss)
            if isinstance(self.memory, PrioritizedReplayBuffer):
                self.memory.update_priorities(indexes, tf.math.abs(target_q_values - current_q_values))
            self.learn_step_counter += 1

    def _update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        network_gradient, _ = tf.clip_by_global_norm(network_gradient, self.max_norm_grad)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def save(self):
        print('Saving models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "wb")
        pickle.dump([self.memory, self.learn_step_counter, self.time_step], f)
        f.close()
        self.q_net.save_weights(self.q_net.checkpoint_file)
        self.q_target_net.save_weights(self.q_target_net.checkpoint_file)

    def load(self):
        print('Loading models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "rb")
        self.memory, self.learn_step_counter, self.time_step = pickle.load(f)
        f.close()
        self.q_net.load_weights(self.q_net.checkpoint_file)
        self.q_target_net.load_weights(self.q_target_net.checkpoint_file)
