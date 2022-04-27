import os
import pickle

from Agents.Agent import Agent
from Agents.TD3.Network import ActorNetwork, CriticNetwork
from Utilities.NetworkBuilder import NetworkBuilder
from Utilities.ReplayBuffer import PrioritizedReplayBuffer
import tensorflow as tf
import numpy as np


class TD3Agent(Agent):
    def __init__(self, environment, learning_rate_actor, learning_rate_critic,
                 loss_actor, loss_critic, update_coeff_target, discount_factor,
                 delay_coeff_actor, noise, warmup, memory_size, memory_alpha,
                 memory_beta, batch_size, network_dict_actor, network_dict_critic,
                 checkpoint_dir='tmp/td3', seed=42):
        super(Agent, self).__init__(environment, batch_size, checkpoint_dir, seed)
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        self.update_coeff_target = update_coeff_target
        self.discount_factor = discount_factor
        self.delay_coeff_actor = delay_coeff_actor
        self.noise = noise
        self.warmup = warmup
        self.memory_size = memory_size
        self.memory_alpha = memory_alpha
        self.memory_beta = memory_beta
        self.network_dict_actor = network_dict_actor
        self.network_dict_critic = network_dict_critic

        self.memory = PrioritizedReplayBuffer(self.memory_size, self.state_space_shape,
                                              self.action_space_shape, self.memory_alpha)

        self.network_builder = NetworkBuilder()
        self.actor = self._init_network(self.network_dict_actor, 'actor', self.checkpoint_dir,
                                        tf.keras.optimizers.Adam, self.learning_rate_actor, self.loss_actor, True)
        self.critic_1 = self._init_network(self.network_dict_critic, 'critic_1', self.checkpoint_dir,
                                           tf.keras.optimizers.Adam, self.learning_rate_critic, self.loss_critic, False)
        self.critic_2 = self._init_network(self.network_dict_critic, 'critic_2', self.checkpoint_dir,
                                           tf.keras.optimizers.Adam, self.learning_rate_critic, self.loss_critic, False)
        self.target_actor = self._init_network(self.network_dict_actor, 'target_actor', self.checkpoint_dir,
                                               tf.keras.optimizers.Adam, self.learning_rate_actor, self.loss_actor, True)
        self.target_critic_1 = self._init_network(self.network_dict_critic, 'target_critic_1', self.checkpoint_dir,
                                                  tf.keras.optimizers.Adam, self.learning_rate_critic, self.loss_critic, False)
        self.target_critic_2 = self._init_network(self.network_dict_critic, 'target_critic_2', self.checkpoint_dir,
                                                  tf.keras.optimizers.Adam, self.learning_rate_critic, self.loss_critic, False)

        self.update_target(1)

    def _init_network(self, params_dict, name, checkpoint_dir, optimizer, learning_rate, loss, actor=True):
        if actor:
            network = ActorNetwork(self.network_builder.build_network(params_dict), name, checkpoint_dir)
        else:
            network = CriticNetwork(self.network_builder.build_network(params_dict), name, checkpoint_dir)
        network.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)
        return network

    def update_target_network(self, network, target_network, update_coeff_target):
        weights = [w * update_coeff_target + target_network.weights[i] * (1 - update_coeff_target) for i, w in
                   enumerate(network.weights)]
        network.set_weights(weights)

    def update_target(self, update_coeff_target=None):
        if update_coeff_target is None:
            update_coeff_target = self.update_coeff_target

        self.update_target_network(self.actor, self.target_actor, update_coeff_target)
        self.update_target_network(self.critic_1, self.target_critic_1, update_coeff_target)
        self.update_target_network(self.critic_2, self.target_critic_2, update_coeff_target)

    def act(self, observation):
        if self.time_step < self.warmup:
            action_prob = np.random.random(self.action_number) * self.action_space_shape
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            action_prob = self.actor(state)[0] + np.random.normal(scale=self.noise)

        action_prob = tf.math.round(tf.clip_by_value(action_prob, 0, self.action_space_shape - 1))
        self.time_step += 1
        return action_prob

    def store(self, initial_state, action, reward, final_state, terminal):
        self.memory.push(initial_state, action, reward, final_state, terminal)

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        initial_states, actions, rewards, final_states, terminals, weights, indexes = self.memory.pop(self.batch_size,
                                                                                                      self.memory_beta)
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(final_states) + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, 0,
                                              self.action_space_shape - 1)  # maybe round the values for discrete space

            q1_ = tf.squeeze(self.target_critic_1(final_states, target_actions), 1)
            q2_ = tf.squeeze(self.target_critic_2(final_states, target_actions), 1)

            q1 = tf.squeeze(self.critic_1(initial_states, actions), 1)
            q2 = tf.squeeze(self.critic_2(initial_states, actions), 1)

            critic_value = tf.math.minimum(q1_, q2_)

            td_target = rewards + self.gamma * critic_value * (1 - terminals)
            td_error = td_target - tf.math.minimum(q1, q2)
            critic_1_loss = tf.keras.losses.MSE(td_target * tf.math.sqrt(weights), q1)
            critic_2_loss = tf.keras.losses.MSE(td_target * tf.math.sqrt(weights), q2)

        self.memory.update_priorities(indexes, td_error)
        self.update_network(self.critic_1, tape, critic_1_loss)
        self.update_network(self.critic_2, tape, critic_2_loss)

        self.learn_step_counter += 1

        if self.learn_step_counter % self.delay_coeff_actor != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(initial_states)
            critic_1_value = self.critic_1(initial_states, new_actions)
            actor_loss = - tf.math.reduce_mean(critic_1_value) * weights
        self.update_network(self.actor, tape, actor_loss)

        self.update_target()

    def update_network(self, network, tape, loss):
        network_gradient = tape.gradient(loss, network.trainable_variables)
        network.optimizer.apply_gradients(zip(network_gradient, network.trainable_variables))

    def save(self):
        print('Saing models and parameters...')
        pickle.dump([self.memory, self.learn_step_counter, self.time_step],
                    open(os.path.join(self.checkpoint_dir, '_params_td3'), "wb"))
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load(self):
        print('Loading models and parameters...')
        self.memory, self.learn_step_counter, self.time_step = pickle.load(
            open(os.path.join(self.checkpoint_dir, '_params_td3'), "rb"))
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)
