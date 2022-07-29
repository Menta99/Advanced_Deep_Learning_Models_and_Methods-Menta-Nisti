import os
import pickle

from Agents.Agent import Agent
from Agents.DDDQN.Network import DQNetwork, DuelingNetwork
from Utilities.NetworkBuilder import NetworkBuilder
from Utilities.ReplayBuffer import PrioritizedReplayBuffer
import tensorflow as tf
import numpy as np


class DDDQNAgent(Agent):
    def __init__(self, environment, q_net_dict, q_target_net_dict, double_q, dueling_q, q_net_update,
                 q_target_net_update, discount_factor, q_net_optimizer, q_target_net_optimizer, q_net_learning_rate,
                 q_target_net_learning_rate, q_net_loss, q_target_net_loss, num_episodes, memory_size, memory_alpha,
                 memory_beta, max_epsilon, min_epsilon, epsilon_A, epsilon_B, epsilon_C, batch_size, checkpoint_dir):
        super(DDDQNAgent, self).__init__(environment, batch_size, checkpoint_dir)
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
        self.memory_size = memory_size
        self.memory_alpha = memory_alpha
        self.memory_beta = memory_beta
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_A = epsilon_A
        self.epsilon_B = epsilon_B
        self.epsilon_C = epsilon_C

        self.memory = PrioritizedReplayBuffer(self.memory_size, self.state_space_shape,
                                              self.action_number, self.memory_alpha)

        self.network_builder = NetworkBuilder()
        self.q_net = self._init_network(self.q_net_dict, 'q_net', self.checkpoint_dir, self.q_net_optimizer,
                                        self.q_net_learning_rate, self.q_net_loss)
        if self.double_q:
            self.q_target_net = self._init_network(self.q_target_net_dict, 'q_target_net', self.checkpoint_dir,
                                                   self.q_target_net_optimizer, self.q_target_net_learning_rate,
                                                   self.q_target_net_loss)
            self._update_network_parameters(self.q_net, self.q_target_net, 1)

    def _update_network_parameters(self, network, target_network, target_update_coefficient):
        weights = [w * target_update_coefficient + target_network.weights[i] * (1 - target_update_coefficient)
                   for i, w in enumerate(network.weights)]
        target_network.set_weights(weights)

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

    def _epsilon_scheduler(self):
        standardized_time = (self.time_step - self.epsilon_A * self.num_episodes) / (self.epsilon_B * self.num_episodes)
        cosh = np.cosh(np.math.exp(-standardized_time))
        epsilon = 1.1 - (1 / cosh + (self.time_step * self.epsilon_C / self.num_episodes))
        return max(min(epsilon, self.max_epsilon), self.min_epsilon)

    def act(self, observation):
        if np.random.random() < self._epsilon_scheduler():
            actions = [np.random.choice(self.action_space_shape[i]) for i in range(self.action_number)]
        else:
            state = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), axis=0)
            if self.dueling_q:
                actions = [np.argmax(tensor, axis=-1) for tensor in self.q_net.advantage(state)]
            else:
                actions = [np.argmax(tensor, axis=-1) for tensor in self.q_net(state)]

        self.time_step += 1
        return actions

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        if self.time_step % self.q_target_net_update == 0 and self.double_q:
            self._update_network_parameters(self.q_net, self.q_target_net, 1)

        if self.time_step % self.q_net_update == 0:
            initial_states, actions, rewards, final_states, terminals, weights, indexes = self.memory.pop(self.batch_size, self.memory_beta)

            if self.double_q:
                q_next = self.q_target_net(final_states)
            else:
                q_next = self.q_net(final_states)
            max_actions = tf.math.argmax(self.q_net(final_states), axis=1)
            q_target = rewards + self.discount_factor * tf.gather(q_next, max_actions, batch_dims=1) * (1 - terminals)

            with tf.GradientTape() as tape:
                q_pred = tf.squeeze(
                    tf.gather(self.q_net(initial_states), tf.cast(actions, dtype=tf.int32), batch_dims=1),
                    axis=1)  # bs,1
                loss = self.q_net_loss(q_target, q_pred, sample_weight=tf.expand_dims(weights, axis=-1))

            self._update_network(self.q_net, tape, loss)
            self.memory.update_priorities_variant(indexes, tf.math.abs(q_target - q_pred))
            self.learn_step_counter += 1

    def _update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def save(self):
        print('Saving models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "wb")
        pickle.dump([self.memory, self.learn_step_counter, self.time_step], f)
        f.close()
        self.q_net.save_weights(self.q_net.checkpoint_file)
        if self.double_q:
            self.q_target_net.save_weights(self.q_target_net.checkpoint_file)

    def load(self):
        print('Loading models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "rb")
        self.memory, self.learn_step_counter, self.time_step = pickle.load(f)
        f.close()
        self.q_net.load_weights(self.q_net.checkpoint_file)
        if self.double_q:
            self.q_target_net.load_weights(self.q_target_net.checkpoint_file)
