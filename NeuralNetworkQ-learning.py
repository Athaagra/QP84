#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:26:21 2022

@author: Optimus
"""
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
            print("Alice has added to the counter a variable")
	
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
            reward = 0
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
            reward = 0
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
	
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random 
import math
import time
import matplotlib.pyplot as plt 
#use_cuda=torch.cuda.is_available()
#device=torch.device("cuda:0" if use_cude else "cpu")
#Tensor = torch.Tensor
#env = gym.make('CartPole-v0')
#seed_value=23
#torch.manual_seed(seed_value)
#random.seed(seed_value)
#learning_rate=0.01
#num_episodes=500
#gamma=0.99
#egreedy=0.9
#egreedy_final=0.02
#egreedy_decay=500
#report_interval=10
#score_to_solve=195

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.autograd import Variable

#if gpu is to be used 
use_cuda=torch.cuda.is_available()
device=torch.device("cuda:0" if use_cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor 
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
number_of_inputs = 4
number_of_outputs = 11
egreedy=0.5
steps=[]
seed_value=0
torch.manual_seed(seed_value)
random.seed(seed_value)
learning_rate=0.01
num_episodes=500
gamma=0.99
egreedy=0.9
egreedy_final=0.02
egreedy_decay=500
report_interval=10
score_to_solve=195

def calculate_epsilon(steps_done):
    epsilon=egreedy_final + (egreedy - egreedy_final) * \
        math.exp(-1. * steps_done / egreedy_decay)
    return epsilon

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,number_of_outputs)
        
    def forward(self, x):
        output = self.linear1(x)
        return output

class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss() 
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
    def select_action(self,state,epsilon):
         random_for_egreedy = torch.rand(1)[0]
         if random_for_egreedy > egreedy:
             with torch.no_grad():
                 state=Tensor(state).to(device)
                 action_from_nn=self.nn(state)
                 action=torch.max(action_from_nn,0)[1]
                 actiona=action.item()
                 action=np.array(actions_list[actiona])
                 print('This is the action {} in egreedy'.format(action))
         else:
            random_values=Q[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            actiona=np.argmax(random_values)
            #random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
            #actionb=np.argmax(random_values)
            action=np.array(actions_list[actiona])
         return action       
    def optimize(self,state,action,new_state,reward,done):
        state=Tensor(state).to(device)
        new_state=Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        #Q[int(state_n[1][0]),actiona]=re + gamma * max(Q[int(stat[1][0])])
        if done:
            target_value = reward
        else:
            new_state_value=self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_value)
            target_value=reward + gamma + max_new_state_values
episodes=100
solved=0
frames_total=0
steps_ep=[]
total_episodes=[]
qnet_agent=QNet_Agent()
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    gamma= 1
    np.random.seed(0)
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    learning_rate=1
    state_n,action,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,len(actions_list))
    print(state_n)
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        frames_total+=1
        epsilon=calculate_epsilon(frames_total)
        action=qnet_agent.select_action(state_n[0],epsilon)
        print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #print(stat[0],state_n[0])
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        qnet_agent.optimize(stat[0],action_h,state_n[0],re,do)
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        #print(Q)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment Neural Network:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Neural Network:{}'.format(solved/episodes))
plt.show()