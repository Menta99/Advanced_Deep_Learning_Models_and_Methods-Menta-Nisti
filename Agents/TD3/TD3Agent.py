import os
import pickle

from Agents.Agent import Agent
from Agents.TD3.Network import ActorNetwork, CriticNetwork
from Utilities.NetworkBuilder import NetworkBuilder
from Utilities.ReplayBuffer import PrioritizedReplayBuffer
import tensorflow as tf
import numpy as np


class TD3Agent(Agent):
    def __init__(self, environment, actor_net_dict, critic_1_net_dict, critic_2_net_dict, actor_target_net_dict,
                 critic_1_target_net_dict, critic_2_target_net_dict, actor_net_update, target_net_update,
                 target_update_coefficient, discount_factor, actor_net_optimizer, critic_1_net_optimizer,
                 critic_2_net_optimizer, actor_target_net_optimizer, critic_1_target_net_optimizer,
                 critic_2_target_net_optimizer, actor_net_learning_rate, critic_1_net_learning_rate,
                 critic_2_net_learning_rate, actor_target_net_learning_rate, critic_1_target_net_learning_rate,
                 critic_2_target_net_learning_rate, actor_net_loss, critic_1_net_loss, critic_2_net_loss,
                 actor_target_net_loss, critic_1_target_net_loss, critic_2_target_net_loss, noise, num_episodes,
                 memory_size, memory_alpha, memory_beta, max_epsilon, min_epsilon, epsilon_A, epsilon_B, epsilon_C,
                 batch_size, checkpoint_dir):
        super(TD3Agent, self).__init__(environment, batch_size, checkpoint_dir)
        self.actor_net_dict = actor_net_dict
        self.critic_1_net_dict = critic_1_net_dict
        self.critic_2_net_dict = critic_2_net_dict
        self.actor_target_net_dict = actor_target_net_dict
        self.critic_1_target_net_dict = critic_1_target_net_dict
        self.critic_2_target_net_dict = critic_2_target_net_dict
        self.actor_net_update = actor_net_update
        self.target_net_update = target_net_update
        self.target_update_coefficient = target_update_coefficient
        self.discount_factor = discount_factor
        self.actor_net_optimizer = actor_net_optimizer
        self.critic_1_net_optimizer = critic_1_net_optimizer
        self.critic_2_net_optimizer = critic_2_net_optimizer
        self.actor_target_net_optimizer = actor_target_net_optimizer
        self.critic_1_target_net_optimizer = critic_1_target_net_optimizer
        self.critic_2_target_net_optimizer = critic_2_target_net_optimizer
        self.actor_net_learning_rate = actor_net_learning_rate
        self.critic_1_net_learning_rate = critic_1_net_learning_rate
        self.critic_2_net_learning_rate = critic_2_net_learning_rate
        self.actor_target_net_learning_rate = actor_target_net_learning_rate
        self.critic_1_target_net_learning_rate = critic_1_target_net_learning_rate
        self.critic_2_target_net_learning_rate = critic_2_target_net_learning_rate
        self.actor_net_loss = actor_net_loss
        self.critic_1_net_loss = critic_1_net_loss
        self.critic_2_net_loss = critic_2_net_loss
        self.actor_target_net_loss = actor_target_net_loss
        self.critic_1_target_net_loss = critic_1_target_net_loss
        self.critic_2_target_net_loss = critic_2_target_net_loss
        self.noise = noise
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
        self.actor_net = self._init_network(self.actor_net_dict, 'actor_net', self.checkpoint_dir,
                                            self.actor_net_optimizer, self.actor_net_learning_rate,
                                            self.actor_net_loss, True)
        self.critic_1_net = self._init_network(self.critic_1_net_dict, 'critic_1_net', self.checkpoint_dir,
                                               self.critic_1_net_optimizer, self.critic_1_net_learning_rate,
                                               self.critic_1_net_loss, False)
        self.critic_2_net = self._init_network(self.critic_2_net_dict, 'critic_2_net', self.checkpoint_dir,
                                               self.critic_2_net_optimizer, self.critic_2_net_learning_rate,
                                               self.critic_2_net_loss, False)
        self.actor_target_net = self._init_network(self.actor_target_net_dict, 'actor_target_net',
                                                   self.checkpoint_dir, self.actor_target_net_optimizer,
                                                   self.actor_target_net_learning_rate, self.actor_target_net_loss,
                                                   True)
        self.critic_1_target_net = self._init_network(self.critic_1_target_net_dict, 'critic_1_target_net',
                                                      self.checkpoint_dir, self.critic_1_target_net_optimizer,
                                                      self.critic_1_target_net_learning_rate,
                                                      self.critic_1_target_net_loss, False)
        self.critic_2_target_net = self._init_network(self.critic_2_target_net_dict, 'critic_2_target_net',
                                                      self.checkpoint_dir, self.critic_2_target_net_optimizer,
                                                      self.critic_2_target_net_learning_rate,
                                                      self.critic_2_target_net_loss, False)
        self._update_target(1)

    def _init_network(self, network_dict, name, checkpoint_dir, optimizer, learning_rate, loss, actor=True):
        if actor:
            base, heads = network_dict
            network = ActorNetwork(self.network_builder.build_network(base),
                                   self.network_builder.build_network(heads),
                                   name, checkpoint_dir)
        else:
            ext, head = network_dict
            network = CriticNetwork(self.network_builder.build_network(ext),
                                    self.network_builder.build_network(head),
                                    name, checkpoint_dir)
        network.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
        return network

    def _update_target(self, coefficient):
        self._update_network_parameters(self.actor_net, self.actor_target_net, coefficient)
        self._update_network_parameters(self.critic_1_net, self.critic_1_target_net, coefficient)
        self._update_network_parameters(self.critic_2_net, self.critic_2_target_net, coefficient)

    def _update_network_parameters(self, network, target_network, coefficient):
        weights = [w * coefficient + target_network.weights[i] * (1 - coefficient)
                   for i, w in enumerate(network.weights)]
        target_network.set_weights(weights)

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
            actions = [np.argmax(action + np.random.normal(scale=self.noise, size=action.shape), axis=-1)
                       for action in self.actor_net(state)]

        self.time_step += 1
        return actions

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        if self.time_step % self.target_net_update == 0:
            self._update_target(self.target_update_coefficient)

        initial_states, actions, rewards, final_states, terminals, weights, indexes = self.memory.pop(self.batch_size,
                                                                                                      self.memory_beta)

        with tf.GradientTape(persistent=True) as tape:
            target_actions_prob = tuple(
                action + tf.clip_by_value(tf.random.normal(shape=action.shape, stddev=self._epsilon_scheduler()), -0.5,
                                          0.5) for
                action in self.actor_target_net(final_states))
            target_actions = tuple(
                tf.expand_dims(self._softargmax(action_prob), axis=-1) for action_prob in target_actions_prob)

            target_critic_1_q_value = tf.squeeze(self.critic_1_target_net(final_states, target_actions), 1)
            target_critic_2_q_value = tf.squeeze(self.critic_2_target_net(final_states, target_actions), 1)

            critic_1_q_value = tf.squeeze(self.critic_1_net(initial_states, (actions,)), 1)
            critic_2_q_value = tf.squeeze(self.critic_2_net(initial_states, (actions,)), 1)

            target_critic_q_value = tf.math.minimum(target_critic_1_q_value, target_critic_2_q_value)

            td_target = rewards + self.discount_factor * target_critic_q_value * (1 - terminals)
            td_error = td_target - tf.math.minimum(critic_1_q_value, critic_2_q_value)
            critic_loss = self.critic_1_net_loss(td_target, critic_1_q_value,
                                                 sample_weight=tf.expand_dims(weights, axis=-1)) + \
                          self.critic_2_net_loss(td_target, critic_2_q_value,
                                                 sample_weight=tf.expand_dims(weights, axis=-1))

        self.memory.update_priorities(indexes, tf.math.abs(td_error))
        self._update_network(self.critic_1_net, tape, critic_loss)
        self._update_network(self.critic_2_net, tape, critic_loss)

        self.learn_step_counter += 1

        if self.time_step % self.actor_net_update == 0:
            with tf.GradientTape() as tape:
                new_actions = tuple(
                    tf.expand_dims(self._softargmax(action_prob), axis=-1) for action_prob in
                    self.actor_net(initial_states))
                critic_1_q_value = tf.squeeze(self.critic_1_net(initial_states, new_actions), 1)
                actor_loss = -tf.math.reduce_mean(critic_1_q_value * weights)
            self._update_network(self.actor_net, tape, actor_loss)

    def _softargmax(self, x, beta=1e10):
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)

    def _update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def save(self):
        print('Saving models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "wb")
        pickle.dump([self.memory, self.learn_step_counter, self.time_step], f)
        f.close()
        self.actor_net.save_weights(self.actor_net.checkpoint_file)
        self.critic_1_net.save_weights(self.critic_1_net.checkpoint_file)
        self.critic_2_net.save_weights(self.critic_2_net.checkpoint_file)
        self.actor_target_net.save_weights(self.actor_target_net.checkpoint_file)
        self.critic_1_target_net.save_weights(self.critic_1_target_net.checkpoint_file)
        self.critic_2_target_net.save_weights(self.critic_2_target_net.checkpoint_file)

    def load(self):
        print('Loading models and parameters...')
        f = open(os.path.join(self.checkpoint_dir, '_params'), "rb")
        self.memory, self.learn_step_counter, self.time_step = pickle.load(f)
        f.close()
        self.actor_net.load_weights(self.actor_net.checkpoint_file)
        self.critic_1_net.load_weights(self.critic_1_net.checkpoint_file)
        self.critic_2_net.load_weights(self.critic_2_net.checkpoint_file)
        self.actor_target_net.load_weights(self.actor_target_net.checkpoint_file)
        self.critic_1_target_net.load_weights(self.critic_1_target_net.checkpoint_file)
        self.critic_2_target_net.load_weights(self.critic_2_target_net.checkpoint_file)
