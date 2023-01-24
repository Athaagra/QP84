#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:15:59 2023

@author: Optimus
"""

"""
Created on Fri Nov 18 00:55:15 2022
@author: Optimus
"""

import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal 
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
class Qprotocol:
     def encoded(data0,q):
         chars="XZ"
         basesender=np.random.randint(0,2,q)
         senderFilter=[chars[i] for i in basesender]
         data1=[]
         for i in range(0,len(data0)):
             if data0[i]==0 and senderFilter[i]=="Z":
                 data1.append('0')
             if data0[i]==1 and senderFilter[i]=="Z":
                 data1.append('1')
             if data0[i]==0 and senderFilter[i]=="X":
                 data1.append('+')
             if data0[i]==1 and senderFilter[i]=="X":
                 data1.append('-')
         return np.array(data1)
     def decoded(data1,q):
         chars="XZ"
         basereciever=np.random.randint(0,2,q)
         recieverFilter=[chars[i] for i in basereciever]
         data2=[]
         for i in range(0,len(data1)):
             if data1[i]=="0" and recieverFilter[i]=="Z":
                 data2.append(0)
             if data1[i]=="1" and recieverFilter[i]=="Z":
                 data2.append(1)
             if data1[i]=="0" and recieverFilter[i]=="X":
                 data2.append(0)
             if data1[i]=="1" and recieverFilter[i]=="X":
                 data2.append(1)
             if data1[i]=="+" and recieverFilter[i]=="X":
                 data2.append(0)
             if data1[i]=="-" and recieverFilter[i]=="X":
                 data2.append(1)
             if data1[i]=="+" and recieverFilter[i]=="Z":
                 data2.append(np.random.randint(0,2,1)[0])
             if data1[i]=="-" and recieverFilter[i]=="Z":
                 data2.append(np.random.randint(0,2,1)[0])
         return np.array(data2)
     def __init__(self,maxm):
         self.max_moves = maxm
         self.data0=np.random.randint(0,2,1)
         self.data1 = np.random.randint(0,2,1)
         self.data2 = np.random.randint(0,2,1)
         # State for alice
         #self.data1 = np.random.randint(0,2,2)
         #self.data2 = np.random.randint(0,2,2)
         self.error_counter=[]
         self.alice_data_counter = 0
         self.alice_datalog = []
         # State for bob
         self.bob_data_counter = 0
         self.bob_has_mail = 0
         self.bob_datalog = -1*np.ones(1)
         self.bob_mailbox = []
         self.bob_key = []
         # General environment properties
         self.done = False
         self.action_history = []
         self.cumulative_reward = 0
         self.alice_observation = -np.ones(self.max_moves)
         self.bob_observation = np.concatenate(([self.bob_has_mail], self.bob_datalog))
         state = (self.alice_observation, self.bob_observation)
         self.state_space = (len(state[0]), len(state[1]))
         self.action_space = (3, 4)
         #self.reset()
     def step(self, action, verbose=0):
         import numpy as np
     # Keep track of action list
         self.action_history.append(action)
         # Reset reward
         reward = 0
         bk=[0]
         # If we have used 10 actions, game over
         if len(self.action_history) > self.max_moves:
             reward = 0
             self.done = True
     # Extract the actions for each player
         action_alice, action_bob = action[0], action[1]
     #------------------------------------------------------------
     # Process actions for alice
     #------------------------------------------------------------
     # Read next bit from data1 to Alice's log
         if( action_alice == 1 ):
             if( self.alice_data_counter >= len(self.data1) ):
                 if verbose:
                     print("Alice tried to read more bits than available")
                 else:
                     self.alice_datalog.append(self.data1[self.alice_data_counter])
                     self.alice_data_counter += 1
                 if verbose:
                     print("Alice added data1 to the datalog ", self.alice_datalog)
     # Send datalog to Bob's mailbox
         if( action_alice == 2 ):
             self.bob_mailbox = self.alice_datalog
             self.bob_has_mail = 1
     #------------------------------------------------------------
     # Process actions for bob
     #------------------------------------------------------------
         if( action_bob == 1 ):
             if self.bob_mailbox:
                 if( self.bob_data_counter >= len(self.bob_mailbox) ):
                     if verbose:
                         print("Bob tried to read more bits than available")
                     else:
                         self.bob_datalog[self.bob_data_counter % len(self.bob_datalog)] = self.bob_mailbox[self.bob_data_counter]
                         self.bob_data_counter += 1
         if verbose:
             print("Bob added to his datalog ", self.bob_datalog)
             # Add 0 to key - Bob should decide to take this action based on his datalog
         if( action_bob == 2 ):
             self.bob_key.append(0)
             #self.bob_key=np.hstack((self.bob_key,0))
                 # reward = 0
             if( len(self.bob_key) == len(self.data2) ):
             # self.done = 
                 #self.bob_key=np.array(self.bob_key)
                 #print('This is data1 {} and data2 {} and Bob key {}'.format(self.data1,self.data2,self.bob_key))
                 a=[self.bob_key[i]==self.data1[i] for i in range(len(self.bob_key))]
                 a=np.array(a)
                 if a.all():
                 #if( np.array(self.bob_key) == self.data2 ):
                     reward = +1
                     #bk=self.bob_key
                     self.cumulative_reward += reward
                     self.done = True
                 else:
                     reward = -1
                     #self.done=True
                     #bk=self.bob_key
                     self.cumulative_reward += reward
                     # Add 1 to key - Bob should decide to take this action based on his datalog
         if( action_bob == 3 ):
             self.bob_key.append(1)
             #self.bob_key=np.hstack((self.bob_key,1))
             # reward = 0
             # If bob wrote enough bits
             if( len(self.bob_key) == len(self.data2) ):
                 # self.done = True
                 #self.bob_key=np.array(self.bob_key)
                 a=[self.bob_key[i]==self.data1[i] for i in range(len(self.bob_key))]
                 a=np.array(a)
                 if a.all():
                 #if( np.array(self.bob_key) == self.data2 ):
                     reward = +1
                     self.cumulative_reward += reward
                     self.done = True
                     #print('This is data1 {} and data2 {} bob key {}'.format(self.data1,self.data2,self.bob_key))

                 else:
                     reward = -1
                     #self.done=True
                     self.cumulative_reward += reward
         # Update the actions that alice and bob took
         self.alice_observation[(len(self.action_history)-1)%len(self.alice_observation)] = action[0]
         self.bob_observation = np.concatenate(([self.bob_has_mail], self.bob_datalog))
         state = (self.alice_observation, self.bob_observation)
         #bk=self.bob_key
         return state, reward, self.done, {'action_history':self.action_history},self.bob_key
     def reset(self,maxm,inputm,encode=encoded,decode=decoded):
         import numpy as np
         self.max_moves = maxm
         # State for alice
         #self.data0=np.random.randint(0,2,2)
         #self.data1 = np.random.randint(0,2,2)
         self.data1=np.array(inputm)
         self.data0=encode(self.data1,len(self.data1))
         #print(self.data0)
         self.data2=decode(self.data0,len(self.data0))
         z=[self.data1[i]==self.data2[i] for i in range(len(self.data1))]
         z=np.array(z)
         if z.all():
             self.error_counter.append(0)
         else:
             self.error_counter.append(1)
         self.alice_data_counter = 0
         self.alice_datalog = []
         # State for bob
         self.bob_data_counter = 0
         self.bob_has_mail = 0
         self.bob_datalog = -1*np.ones(1)
         self.bob_mailbox = []
         self.bob_key = []#np.array([])#[]
         # General environment properties
         self.done = False
         self.action_history = []
         self.cumulative_reward = 0
         self.alice_observation = -np.ones(self.max_moves)
         self.bob_observation = np.concatenate(([self.bob_has_mail], self.bob_datalog))
         state = (self.alice_observation, self.bob_observation)
         self.state_space = (len(state[0]), len(state[1]))
         self.action_space = (3, 4)
         return state
     def render(self):
         print("---Alice---")
         print("- Datalog: ", self.alice_datalog)
         print("---Bob---")
         print("- Has Mail: ", self.bob_has_mail)
         print("- Datalog: ", self.bob_datalog)
         print("")
         return


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

@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action1(observation):
    logits = actor1(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


@tf.function
def sample_action2(observation):
    logits = actor2(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action3(observation):
    logits = actor3(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action4(observation):
    logits = actor4(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action5(observation):
    logits = actor5(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action6(observation):
    logits = actor6(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action7(observation):
    logits = actor7(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action8(observation):
    logits = actor8(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action9(observation):
    logits = actor9(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action10(observation):
    logits = actor10(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action11(observation):
    logits = actor11(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action12(observation):
    logits = actor12(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action13(observation):
    logits = actor13(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action14(observation):
    logits = actor14(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def sample_action15(observation):
    logits = actor15(observation)
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
observation_dimensions = 4
num_actions = 12

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


actor2 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic2 = keras.Model(inputs=observation_input, outputs=value)
actor3 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic3 = keras.Model(inputs=observation_input, outputs=value)


actor4 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic4 = keras.Model(inputs=observation_input, outputs=value)
actor5 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic5 = keras.Model(inputs=observation_input, outputs=value)


actor6 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic6 = keras.Model(inputs=observation_input, outputs=value)
actor7 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic7 = keras.Model(inputs=observation_input, outputs=value)

actor8 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic8 = keras.Model(inputs=observation_input, outputs=value)
actor9 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic9 = keras.Model(inputs=observation_input, outputs=value)

actor10 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic10 = keras.Model(inputs=observation_input, outputs=value)
actor11 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic11 = keras.Model(inputs=observation_input, outputs=value)


actor12 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic12 = keras.Model(inputs=observation_input, outputs=value)
actor13 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic13 = keras.Model(inputs=observation_input, outputs=value)

actor14 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic14 = keras.Model(inputs=observation_input, outputs=value)
actor15 = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    #mlp(observation_input,[1], tf.tanh, None), axis=1
)
critic15 = keras.Model(inputs=observation_input, outputs=value)


# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rewards_during_training=[]
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
# Iterate over the number of epochs
LogicalStates=np.array([[1,0],[0,1]])
LogicalStates2bit=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
LogicalStates3bit=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
LogicalStates4bit=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
import pandas as pd
columns2bit=['00','01','10','11']
columns3bit=['000','001','010','011','100','101','110','111']
columns4bit=['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
LogicalStates2bit=pd.DataFrame(LogicalStates2bit, columns=columns2bit)
LogicalStates3bit=pd.DataFrame(LogicalStates3bit, columns=columns3bit)
LogicalStates4bit=pd.DataFrame(LogicalStates4bit, columns=columns4bit)
LogicalStates2bit=LogicalStates2bit.rename(index={0:'00',1:'01',2:'10',3:'11'})
LogicalStates3bit=LogicalStates3bit.rename(index={0:'000',1:'001',2:'010',3:'011',4:'100',5:'101',6:'110',7:'111'})
LogicalStates4bit=LogicalStates4bit.rename(index={0:'0000',1:'0001',2:'0010',3:'0011',4:'0100',5:'0101',6:'0110',7:'0111',8:'1000',9:'1001',10:'1010',11:'1011',12:'1100',13:'1101',14:'1110',15:'1111'})
def proximalpo(inp):
    inpu=inp
    q_value_critic=[]
    action_actor=[]
    env=Qprotocol(4)
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done=False
        state=env.reset(4,inpu)
        actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        observation = state[0]
        episode_return=0
        episode_length=0
        while done != True:
        # Iterate over the steps of each epoch
            if render:
                print(render)
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation)
            actiona=np.array(actiona)
            log=np.array(logits[0])
            actiona=actiona[0]
            action_actor.append(log[actiona])
            action=np.array(actions_list[actiona])
            new_state,reward,done,info,bob_key=env.step(action)
            episode_return += reward
            episode_length += 1        
            # Get the value and log-probability of the action
            value_t = critic(observation)
            q_value_critic.append(value_t)
            logprobability_t = logprobabilities(logits, actiona)
            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, actiona, reward, value_t, logprobability_t)
            # Update the observation
            observation = np.array(new_state[0])
            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal: #or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                state=env.reset(4,inpu)
                observation=np.array(state[0]) 
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
    def save_weights(inpt,maxm):
        path= '/home/Optimus/Desktop/QuantumComputingThesis/'
        actor.save(path+ '_actor'+str(maxm)+'One'+str(inpt)+'.h5')
        critic.save(path+ '_critic'+str(maxm)+'One'+str(inpt)+'.h5')
    count=0
    for i in rewards_during_training:
        if i==1.0:
            count+=1
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
    save_weights(inpu,4)
    return actor,critic

def onebitsimulation(inp,ac,cr):
    total_episodes=[]
    solved=0
    episodes=100
    env=Qprotocol(4)
    env1=Qprotocol(4)
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        # reset environment and episode reward
        state=env.reset(inp,4)
        state1=env1.reset(inp,4)
        ep_reward = 0
        done1=False
        done2=False
        observation1 = state[0]
        observation2 = state1[0]
        steps_ep=1
        while done1!=True or done2!=True:
            print('This is the episode {}'.format(i_episode))
            observation1 = observation1.reshape(1, -1)
            observation2 = observation2.reshape(1, -1)
            logits, actiona = sample_action(observation1)
            logits, actionb = sample_action1(observation2)
            actiona=actiona[0]
            actionb=actionb[0]
            actionAA=np.array(actions_list[actiona])
            actionBB=np.array(actions_list[actionb])
            stat1,reward1,done1,action_h1,bob_key1=env.step(actionAA)
            stat2,reward2,done2,action_h2,bob_key2=env1.step(actionBB)
            observation1=stat1[0]
            observation2=stat2[0]
            steps_ep+=1
            if done1:
                bob_key=bob_key1
            if done2:
                bob_key=bob_key2
            if done1==True or done2==True:
                if len(inp)==1 and len(bob_key)==len(inp):
                    tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                if reward1>0 or reward2>0 :
                    count+=1
                    steps_epi.append(steps_ep)
                    if reward1==1 or reward2==1:
                        r=1
                        solved+=1
                    else:
                        r=-1
                    print(r)
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
        
    
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(solved/episodes))
    plt.show()
    
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(np.max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()


actor,critic=proximalpo([0])
actor1,criti1c=proximalpo([1])
onebitsimulation(np.random.ramdint(0,2,1),actor,actor1)

def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic.load_weights(path+ '_criticOne0.h5')
    actor.load_weights(path+ '_actorOne0.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne1.h5')
    actor1.load_weights(path+ '_actorOne1.h5')





def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic.load_weights(path+ '_criticOne00.h5')
    actor.load_weights(path+ '_actorOne00.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne01.h5')
    actor1.load_weights(path+ '_actorOne01.h5')
def load_weightsThree():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic2.load_weights(path+ '_criticOne10.h5')
    actor2.load_weights(path+ '_actorOne10.h5')
def load_weightsFour():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic3.load_weights(path+ '_criticOne11.h5')
    actor3.load_weights(path+ '_actorOne11.h5')
    

def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne000.h5')
    actor1.load_weights(path+ '_actorOne000.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic2.load_weights(path+ '_criticOne001.h5')
    actor2.load_weights(path+ '_actorOne001.h5')
def load_weightsThree():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic3.load_weights(path+ '_criticOne010.h5')
    actor3.load_weights(path+ '_actorOne010.h5')
def load_weightsFour():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic4.load_weights(path+ '_criticOne011.h5')
    actor4.load_weights(path+ '_actorOne011.h5')
def load_weightsFive():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic5.load_weights(path+ '_criticOne100.h5')
    actor5.load_weights(path+ '_actorOne100.h5')
def load_weightsSix():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic6.load_weights(path+ '_criticOne101.h5')
    actor6.load_weights(path+ '_actorOne101.h5')
def load_weightsSeven():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic7.load_weights(path+ '_criticOne110.h5')
    actor7.load_weights(path+ '_actorOne110.h5')
def load_weightsEight():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic8.load_weights(path+ '_criticOne111.h5')
    actor8.load_weights(path+ '_actorOne111.h5')


#def load_weightsOne():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic1.load_weights(path+ '_criticOne0000.h5')
#    actor1.load_weights(path+ '_actorOne0000.h5')
#def load_weightsTwo():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic2.load_weights(path+ '_criticOne0001.h5')
#    actor2.load_weights(path+ '_actorOne0001.h5')
#def load_weightsThree():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic3.load_weights(path+ '_criticOne0010.h5')
#    actor3.load_weights(path+ '_actorOne0010.h5')
#def load_weightsFour():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic4.load_weights(path+ '_criticOne0011.h5')
#    actor4.load_weights(path+ '_actorOne0011.h5')
#def load_weightsFive():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic5.load_weights(path+ '_criticOne0100.h5')
#    actor5.load_weights(path+ '_actorOne0100.h5')
#def load_weightsSix():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic6.load_weights(path+ '_criticOne0101.h5')
#    actor6.load_weights(path+ '_actorOne0101.h5')
#def load_weightsSeven():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic7.load_weights(path+ '_criticOne0110.h5')
#    actor7.load_weights(path+ '_actorOne0110.h5')
#def load_weightsEight():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic8.load_weights(path+ '_criticOne0111.h5')
#    actor8.load_weights(path+ '_actorOne0111.h5')
#def load_weightsNine():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic9.load_weights(path+ '_criticOne1000.h5')
#    actor9.load_weights(path+ '_actorOne1000.h5')
#def load_weightsTen():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic10.load_weights(path+ '_criticOne1001.h5')
#    actor10.load_weights(path+ '_actorOne1001.h5')
#def load_weightsEleven():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic11.load_weights(path+ '_criticOne1010.h5')
#    actor11.load_weights(path+ '_actorOne1010.h5')
#def load_weightsTwelve():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic12.load_weights(path+ '_criticOne1011.h5')
#    actor12.load_weights(path+ '_actorOne1011.h5')
#def load_weightsThirteen():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic13.load_weights(path+ '_criticOne1101.h5')
#    actor13.load_weights(path+ '_actorOne1101.h5')
#def load_weightsFourteen():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic14.load_weights(path+ '_criticOne1110.h5')
#    actor14.load_weights(path+ '_actorOne1110.h5')
#def load_weightsFifteen():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic15.load_weights(path+ '_criticOne1111.h5')
#    actor15.load_weights(path+ '_actorOne1111.h5')
#save_weights()
#load_weightsOne()
#load_weightsTwo()



#load_weightsOne()
#load_weightsTwo()
#load_weightsThree()
#load_weightsFour()


load_weightsOne()
load_weightsTwo()
load_weightsThree()
load_weightsFour()
load_weightsFive()
load_weightsSix()
load_weightsSeven()
load_weightsEight()

#load_weightsOne()
#load_weightsTwo()
#load_weightsThree()
#load_weightsFour()
#load_weightsFive()
#load_weightsSix()
#load_weightsEight()
#load_weightsNine()
#load_weightsTen()
#load_weightsEleven()
#load_weightsTwelve()
#load_weightsThirteen()
#load_weightsFourteen()
#load_weightsFifteen()



total_episodes=[]
solved=0
episodes=100
steps_epi=[]
cum_rev=0
cumulative_reward=[]
r=0
count=0
# run infinitely many episodes
for i_episode in range(episodes):
    # reset environment and episode reward
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
    ep_reward = 0
    done1=False
    done2=False
    done3=False
    done4=False
    done5=False
    done6=False
    done7=False
    done8=False
    #done9=False
    #done10=False
    #done11=False
    #done12=False
    #done13=False
    #done14=False
    #done15=False
    #done16=False
    observation1 = state_n[0]
    observation2 = state_n[0]
    observation3 = state_n[0]
    observation4 = state_n[0]
    observation5 = state_n[0]
    observation6 = state_n[0]
    observation7 = state_n[0]
    observation8 = state_n[0]
    #observation9 = state_n[0]
    #observation10 = state_n[0]
    #observation11 = state_n[0]
    #observation12 = state_n[0]
    #observation13 = state_n[0]
    #observation14 = state_n[0]
    #observation15 = state_n[0]
    #observation16 = state_n[0]
    steps_ep=1
    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    while done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True:# or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
        # Get the logits, action, and take one step in the environment
        #observation=observation[0]
        #if len(observation)==2:
        #    observation=observation[0]
        #print('Observation shape {}'.format(observation))
        print('This is the episode {}'.format(i_episode))
        observation1 = observation1.reshape(1, -1)
        observation2 = observation2.reshape(1, -1)
        observation3 = observation3.reshape(1, -1)
        observation4 = observation4.reshape(1, -1)
        observation5 = observation5.reshape(1, -1)
        observation6 = observation6.reshape(1, -1)
        observation7 = observation7.reshape(1, -1)
        observation8 = observation8.reshape(1, -1)
        #observation9 = observation9.reshape(1, -1)
        #observation10 = observation10.reshape(1, -1)
        #observation11 = observation11.reshape(1, -1)
        #observation12 = observation12.reshape(1, -1)
        #observation13 = observation13.reshape(1, -1)
        #observation14 = observation14.reshape(1, -1)
        #observation15 = observation15.reshape(1, -1)
        #observation16 = observation16.reshape(1, -1)
        #print('Observation shape {}'.format(observation))
        #observation=observation[0]
        logits, actiona = sample_action(observation1)
        logits, actionb = sample_action1(observation2)
        logits, actionc = sample_action2(observation3)
        logits, actiond = sample_action3(observation4)
        logits, actione = sample_action4(observation5)
        logits, actionf = sample_action5(observation6)
        logits, actiong = sample_action6(observation7)
        logits, actionh = sample_action7(observation8)
        #logits, actioni = sample_action8(observation9)
        #logits, actionk = sample_action9(observation10)
        #logits, actionl = sample_action10(observation11)
        #logits, actionm = sample_action11(observation12)
        #logits, actionn = sample_action12(observation13)
        #logits, actiono = sample_action13(observation14)
        #logits, actionp = sample_action14(observation15)
        #logits, actionq = sample_action15(observation16)
        actiona=actiona[0]
        actionb=actionb[0]
        actionc=actionc[0]
        actiond=actiond[0]
        actione=actione[0]
        actionf=actionf[0]
        actiong=actiong[0]
        actionh=actionh[0]
        #actioni=actioni[0]
        #actionk=actionk[0]
        #actionl=actionl[0]
        #actionm=actionm[0]
        #actionn=actionn[0]
        #actiono=actiono[0]
        #actionp=actionp[0]
        ##actionq=actionq[0]
        #actionr=actionr[0]
        #actions=actions[0]
        #print('This are logits {} and actions {}'.format(logits, actiona))
        actionAA=np.array(actions_list[actiona])
        actionBB=np.array(actions_list[actionb])
        actionCC=np.array(actions_list[actionc])
        actionDD=np.array(actions_list[actiond])
        actionEE=np.array(actions_list[actione])
        actionFF=np.array(actions_list[actionf])
        actionGG=np.array(actions_list[actiong])
        actionHH=np.array(actions_list[actionh])
        #actionII=np.array(actions_list[actioni])
        #actionKK=np.array(actions_list[actionk])
        #actionLL=np.array(actions_list[actionl])
        #actionMM=np.array(actions_list[actionm])
        #actionNN=np.array(actions_list[actionn])
        #actionOO=np.array(actions_list[actiono])
        #actionPP=np.array(actions_list[actionp])
        #actionQQ=np.array(actions_list[actionq])
        #print('This is the action {}'.format(action))
        stat1,re1,do1,action_h1,bob_key1=step(actionAA,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done1,dedoce, verbose=0,)
        stat2,re2,do2,action_h2,bob_key2=step(actionBB,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done2,dedoce, verbose=0,)
        stat3,re3,do3,action_h3,bob_key3=step(actionCC,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done3,dedoce, verbose=0,)
        stat4,re4,do4,action_h4,bob_key4=step(actionDD,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done4,dedoce, verbose=0,)
        stat5,re5,do5,action_h5,bob_key5=step(actionEE,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done5,dedoce, verbose=0,)
        stat6,re6,do6,action_h6,bob_key6=step(actionFF,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done6,dedoce, verbose=0,)
        stat7,re7,do7,action_h7,bob_key7=step(actionGG,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done7,dedoce, verbose=0,)
        stat8,re8,do8,action_h8,bob_key8=step(actionHH,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done8,dedoce, verbose=0,)
        #stat9,re9,do9,action_h9,bob_key9=step(actionII,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done9,dedoce, verbose=0,)
        #stat10,re10,do10,action_h10,bob_key10=step(actionKK,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done10,dedoce, verbose=0,)
        #stat11,re11,do11,action_h11,bob_key11=step(actionLL,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done11,dedoce, verbose=0,)
        #stat12,re12,do12,action_h12,bob_key12=step(actionMM,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done12,dedoce, verbose=0,)
        #stat13,re13,do13,action_h13,bob_key13=step(actionNN,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done13,dedoce, verbose=0,)
        #stat14,re14,do14,action_h14,bob_key14=step(actionOO,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done14,dedoce, verbose=0,)
        #stat15,re15,do15,action_h15,bob_key15=step(actionPP,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done15,dedoce, verbose=0,)
        ##stat16,re16,do16,action_h16,bob_key16=step(actionQQ,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done16,dedoce, verbose=0,)
        #observation_new, reward, done, _,op = env.step(action[0].numpy())
        done1=do1
        done2=do2
        done3=do3
        done4=do4
        done5=do5
        done6=do6
        done7=do7
        done8=do8
        #done9=do9
        #done10=do10
        #done11=do11
        #done12=do12
        #done13=do13
        #done14=do14
        #done15=do15
        #done16=do16
        observation1=stat1[0]
        observation2=stat2[0]
        observation3=stat3[0]
        observation4=stat4[0]
        observation5=stat5[0]
        observation6=stat6[0]
        observation7=stat7[0]
        observation8=stat8[0]
        #observation8=stat9[0]
        #observation9=stat10[0]
        #observation10=stat11[0]
        #observation11=stat12[0]
        #observation12=stat13[0]
        #observation13=stat14[0]
        #observation14=stat15[0]
        #observation15=stat16[0]
        #print('This is the reward {},{},{},{}'.format(re1,re2,re3,re4))
        #episode_return += re
        steps_ep+=1
        #episode_length += 1
        #observation=stat[0]
        if do1==True or do2==True or do3==True or do4==True or do5==True or do6==True or do7==True or do8==True:# or do9==True or do10==True or do11==True or do12==True or do13==True or do14==True or do15==True or do16==True: # or do9==True:
            #if re1>0 or re2>0 or re3>0 or re4>0 or re5>0 or re6>0 or re7>0 or re8>0: #or re9>0:#do1==True or do2==True or do3==True or do4==True:
                count+=1
                steps_epi.append(steps_ep)
                if re1==1 or re2==1 or re3==1 or re4==1 or re5==1 or re6==1 or re7==1 or re8==1:# or re9==1 or re10==1 or re11==1 or re12==1 or re13==1 or re14==1 or re15==1 or re16==1:# or re9==1 or re10==1 or re11==1 or re12==1 or re13==1 or re14==1 or re15==1 or re16==1: # or re9==1:
                    r=1
                    solved+=1
                else:
                    r=-1
                print(r)
                cum_rev+=r
                #print('This is the reward {}'.format(r))
                #print(cum_re)
                cumulative_reward.append(cum_rev)
                total_episodes.append(r)
                break
    

plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(np.max(cumulative_reward)))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_epi)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.title('The number of steps:{}'.format(np.average(steps_epi)))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()
