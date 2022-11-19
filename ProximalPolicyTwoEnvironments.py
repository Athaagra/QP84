#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 00:55:15 2022

@author: Optimus
"""

import argparse
import gym
import numpy as np
from itertools import count
#import matplotlib.pyplot.plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#import torch
#import torch.nn as nn
import numpy as np
#import torch.optim as optim
import gym
import random
import math
import time
import matplotlib.pyplot as plt

#env = gym.make("CartPole-v1")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from ple import PLE
#from ple.games.flappybird import FlappyBird
import sys

"""
The environment for Level1
	
Actions for Alice:
0 - Idle
1 - Read next bit from data1, store in datalog
2 - Place datalog in Bob's mailbox

Actions for Bob:
0 - Idle
1 - Read next bit from mailbox
2 - Write 0 to key
3 - Write 1 to key
	
Actions are input to the environment as tuples
e.g. (1,0) means Alice takes action 1 and Bob takes action 0
	
Rewards accumulate: negative points for wrong guess, 
positive points for correct guess
Game terminates with correct key or N moves

"""
def render(alice_datalog,bob_datalog,bob_has_mail):
    print("---Alice---")
    print("- Datalog: ", alice_datalog)
    print("---Bob---")
    print("- Has Mail: ", bob_has_mail)
    print("- Datalog: ", bob_datalog)
    print("")
    return
    

def step(action,action_history,max_moves, alice_data_counter,data1,alice_datalog,bob_data_counter,alice_observation,bob_key,bob_datalog,cumulative_reward,bob_mailbox,bob_has_mail,done, verbose=0,):
	
    # Keep track of action list
    action_history.append(action)
	
    # Reset reward
    reward = 0
	
    # If we have used 10 actions, game over
    if len(action_history) > max_moves:
        reward = 0
        done = True
	
    # Extract the actions for each player
    action_alice, action_bob = action[0], action[1]

#------------------------------------------------------------
# Process actions for alice
#------------------------------------------------------------
# Read next bit from data1 to Alice's log
    if( action_alice == 1 ):
        if( alice_data_counter >= len(data1) ):
            if verbose:
             print("Alice tried to read more bits than available")
        else:
            alice_datalog.append(data1[alice_data_counter])
            alice_data_counter += 1
            #print("Alice has added to the counter a variable")
	
    if verbose:
        print("Alice added data1 to the datalog ", alice_datalog)
	
    # Send datalog to Bob's mailbox
    if( action_alice == 2 ):
        bob_mailbox = alice_datalog
        bob_has_mail = 1	
#------------------------------------------------------------
# Process actions for bob
#------------------------------------------------------------
	
    if( action_bob == 1 ):
        if bob_mailbox:
            if( bob_data_counter >= len(bob_mailbox) ):
                if verbose:
                    print("Bob tried to read more bits than available")
            else:
                bob_datalog[bob_data_counter % len(bob_datalog)] = bob_mailbox[bob_data_counter]
                bob_data_counter += 1
	
    if verbose:
        print("Bob added to his datalog ", bob_datalog)
	
    # Add 0 to key - Bob should decide to take this action based on his datalog
    if( action_bob == 2 ):
        bob_key.append(0)
	
    # reward = 0
    if( len(bob_key) == len(data1) ):
        done = True
        if( np.array(bob_key).all() == data1.all() ):
            reward = +1
            cumulative_reward += reward
            done = True
        else:
            reward = -1
            cumulative_reward += reward
		
    # Add 1 to key - Bob should decide to take this action based on his datalog
    if( action_bob == 3 ):
            bob_key.append(1)
	
    # reward = 0
    # If bob wrote enough bits
    if( len(bob_key) == len(data1) ):
        done = True
        if( np.array(bob_key).all() == data1.all() ):
            reward = +1
            cumulative_reward += reward
            done = True
        else:
            reward = -1
            cumulative_reward += reward
	
    # Update the actions that alice and bob took
    alice_observation[(len(action_history)-1)%len(alice_observation)] = action[0]
    bob_observation = np.concatenate(([bob_has_mail], bob_datalog))
    state = (alice_observation, bob_observation)
    #render(alice_observation,bob_datalog, bob_has_mail)
	
    return state, reward, done, {'action_history':action_history},bob_key
	
def reset():
    max_moves = 4
	
    # State for alice
    data1 = np.random.randint(0,2,1)
    #data1=np.array([1])
    alice_data_counter = 0
    alice_datalog = []
	
    # State for bob
    bob_data_counter = 0
    bob_has_mail = 0
    bob_datalog = -1*np.ones(1)
    bob_mailbox = []
    bob_key = []
	
    # General environment properties
    done = False
    action_history = []
    cumulative_reward = 0
	
    alice_observation = -np.ones(max_moves)#self.max_moves)
    bob_observation = np.concatenate(([bob_has_mail], bob_datalog))
    state = (alice_observation, bob_observation)	
    state_space = (len(state[0]), len(state[1]))
    action_space = (3, 4)
    actions=(0,1)
    return state,actions,data1,alice_data_counter,alice_datalog,bob_data_counter,bob_has_mail,bob_mailbox,bob_key,done,action_history,cumulative_reward,state_space,action_space,max_moves,alice_observation,bob_datalog 
	

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import gym 
import scipy.signal 
import time 

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        #print('This is the buffer')
        #print('This is the observation {}'.format(observation))
        #print('This is the action {}'.format(action))
        #print('This is the reward {}'.format(reward))
        #print('This is the value {}'.format(value))
        #print('This is the logprobability {}'.format(logprobability))
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        print(size)
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    print('This is the logprobabilities all {}'.format(logprobabilities_all))
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability
#def _build_model():
#    model = Sequential()
    #24
#    model.add(Dense(12, input_dim=4, activation='softmax'))
    #model.add(Dense(24, activation='relu'))
    #model.add(Dense(self.action_size, activation='linear'))
    #model.add(Dense(self.action_size, activation=''))
#    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=Adam(lr=0.25))
#    return model 

# Sample action from actor
#modela=_build_model()
@tf.function
def sample_action(observation):
    logits = actor(observation)
    #logits = model.predict(observation)
    #logits=np.array(logits)
    #print(logits)
    #print('This are the logits {}'.format(tf.squeeze(logits, 1), axis=1))
    #print('This is the action {}'.format(action))
    #print('This is the categorical {}'.format(tf.squeeze(tf.random.categorical(logits, 1), axis=1)))
    #action = np.argmax(logits[0])
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action1(observation):
    logits = actor1(observation)
    #logits = model.predict(observation)
    #logits=np.array(logits)
    #print(logits)
    #print('This are the logits {}'.format(tf.squeeze(logits, 1), axis=1))
    #print('This is the action {}'.format(action))
    #print('This is the categorical {}'.format(tf.squeeze(tf.random.categorical(logits, 1), axis=1)))
    #action = np.argmax(logits[0])
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):
    print('This is the train policy')
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            #(observation_buffer)
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        print('This is the ratio {} advantage buffer'.format(ratio))
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        print('This is the min_advantage {} 1+ clip_ratio*advantage_buffer, 1-clip_ratio*advantage_buffer '.format(min_advantage))
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
        print('This is the policy grads {} ratio * advantage buffer , minimum advantage'.format(policy_loss))
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    print('Policy grads {}'.format(policy_grads))
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
    print('policy optimizer {}'.format(policy_optimizer))
    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    print('This is the kl {} logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer)'.format(kl))
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    print('This is the observation_buffer {} and the return buffer {}'.format(observation_buffer, return_buffer))
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        print('This is the tape {}'.format(tape))
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
        print('This is the value loss {} return - critic(observation_buffer)) ** 2'.format(value_loss))
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    print('This is the value_grads {} value_loss, critic.trainbable_variables'.format(value_grads))
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Hyperparameters of the PPO algorithm
steps_per_epoch = 100
epochs = 100
gamma = 0.99
clip_ratio = 0.01
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 5000
train_value_iterations = 5000
lam = 0.97
target_kl = 0.01
hidden_sizes = (16, 16)

# True if you want to render the environment
render = False

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
#env = gym.make("CartPole-v0")
observation_dimensions = 4#env.observation_space.shape[0]
num_actions = 12#env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)
actor1 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic1 = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rewards_during_training=[]
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
# Iterate over the number of epochs
q_value_critic=[]
action_actor=[]
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    do=False
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    #observation=np.array(state_n[0])
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    observation = state_n[0]#np.concatenate((state_n[0], state_n[1]), axis=None)
    episode_return=0
    episode_length=0
    while do != True:
    # Iterate over the steps of each epoch
    #for t in range(steps_per_epoch):
        if render:
            print(render)
            #env.render()

        # Get the logits, action, and take one step in the environment
        #observation=observation[0]
        #if len(observation)==2:
        #    observation=observation[0]
        #print('Observation shape {}'.format(observation))
        #print(type(observation))
        observation = observation.reshape(1, -1)
        #print('Observation shape {}'.format(observation))
        #observation=observation[0]
        logits, actiona = sample_action(observation)
        actiona=np.array(actiona)
        print('this is the action {}'.format(actiona))
        print('logits {}'.format(logits))
        log=np.array(logits[0])
        actiona=actiona[0]
        action_actor.append(log[actiona])
        #actiona=actiona[0]
        #print(actiona)
        #print(np.argmax(logits[0]))
        #actiona = np.argmax(logits[0])
        #print('This are logits {} and actions {}'.format(logits, actiona))
        action=np.array(actions_list[actiona])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #observation_new, reward, done, _,op = env.step(action[0].numpy())
        episode_return += re
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        q_value_critic.append(value_t)
        logprobability_t = logprobabilities(logits, actiona)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, actiona, re, value_t, logprobability_t)

        # Update the observation
        observation = np.array(stat[0])

        # Finish trajectory if reached to a terminal state
        terminal = do
        done=do
        if terminal: #or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            #print(episode_return)
            sum_length += episode_length
            num_episodes += 1
            state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()

            observation=np.array(state_n[0]) 
            episode_return=0
            episode_length = 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break
#    print('This is the kl {}'.format(kl))
    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)
    rewards_during_training.append(sum_return / num_episodes)
    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
def save_weights():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    actor.save(path+ '_actorOne.h5')
    critic.save(path+ '_criticOne.h5')
def load_weights():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic.load_weights(path+ '_critic.h5')
    actor.load_weights(path+ '_actor.h5')
def load_weights():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne.h5')
    actor1.load_weights(path+ '_actorOne.h5')
save_weights()
load_weights()
count=0
for i in rewards_during_training:
    if i==1.0:
        count+=1
#save_weights()
#load_weights()
plt.figure(figsize=(13, 13))
plt.plot(rewards_during_training)
plt.xlabel(f'Number of episodes')
plt.ylabel('Average Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy:{}'.format(count))#.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(q_value_critic)
plt.plot(action_actor)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation')
plt.show()




total_episodes=[]
solved=0
episodes=100
# run infinitely many episodes
for i_episode in range(episodes):
    # reset environment and episode reward
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    ep_reward = 0
    done1=False
    done2=False
    observation = state_n[0]
    observation1 = state_n[0]
    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    while done1!=True or done2!=True:
        # Get the logits, action, and take one step in the environment
        #observation=observation[0]
        #if len(observation)==2:
        #    observation=observation[0]
        #print('Observation shape {}'.format(observation))
        #print(type(observation))
        observation = observation.reshape(1, -1)
        observation1 = observation1.reshape(1, -1)
        #print('Observation shape {}'.format(observation))
        #observation=observation[0]
        logits, actiona = sample_action(observation)
        logits, actionb = sample_action1(observation1)
        actiona=actiona[0]
        actionb=actionb[0]
        #print('This are logits {} and actions {}'.format(logits, actiona))
        actionZ=np.array(actions_list[actiona])
        actionO=np.array(actions_list[actionb])
        #print('This is the action {}'.format(action))
        stat1,re1,do1,action_h,bob_key=step(actionZ,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        stat2,re2,do2,action_h,bob_key=step(actionO,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #observation_new, reward, done, _,op = env.step(action[0].numpy())
        done1=do1
        done2=do2
        observation=stat1[0]
        observation1=stat2[0]
        print('This is the reward {}'.format(re))
        episode_return += re
        episode_length += 1
        observation=stat[0]
        if done1==True or done2==True:
            total_episodes.append(re)
        if re==1:
            solved+=1
    

plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(solved/episodes))
plt.show()
