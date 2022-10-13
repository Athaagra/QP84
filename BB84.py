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
	
episodes=100
solved=0
steps_ep=[]
total_episodes=[]
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    gamma= 1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,4)
    print(state_n)
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q[int(state_n[1][0])] + np.random.randint(2, size=(1,max_moves))/1000
        actiona=np.argmax(random_values)
        random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
        actionb=np.argmax(random_values)
        action=(actiona,actionb)
        print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q[int(state_n[1][0]),actiona]=re + gamma * max(Q[int(stat[1][0])])
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        #print(Q)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1 
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Belman Equation:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.title('The simulation has been solved the environment Bellman Equation:{}'.format(solved/episodes))
plt.show()

episodes=100
solved=0
steps_ep=[]
total_episodes=[]
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    gamma= 1
    learning_rate=1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,4)
    print(state_n)
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q[int(state_n[1][0])] + np.random.randint(2, size=(1,max_moves))/1000
        actiona=np.argmax(random_values)
        random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
        actionb=np.argmax(random_values)
        action=(actiona,actionb)
        print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q[int(state_n[1][0]),actiona]=(1-learning_rate)*Q[int(state_n[1][0]),actiona]+learning_rate * (re + gamma * max(Q[int(stat[1][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        #print(Q)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


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

learning_rate=0.01

number_of_inputs = 4
number_of_outputs = 4
egreedy=0.5
steps=[]

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
    def select_action(self,state):
         random_for_egreedy = torch.rand(1)[0]
         if random_for_egreedy > egreedy:
             with torch.no_grad():
                 state=Tensor(state).to(device)
                 action_from_nn=self.nn(state)
                 action=torch.max(action_from_nn,0)[1]
                 action=action.item()
         else:
            random_values=Q[int(state_n[1][0])] + np.random.randint(2, size=(1,max_moves))/1000
            actiona=np.argmax(random_values)
            random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
            actionb=np.argmax(random_values)
            action=(actiona,actionb)
         return action       
    def optimize(self,state,action,new_state,reward,done):
        state=Tensor(state).to(device)
        new_state=Tensor(new_state).to(device)
        reward = Tensor(reward+1).to(device)
        #Q[int(state_n[1][0]),actiona]=re + gamma * max(Q[int(stat[1][0])])
        if done:
            target_value = reward
        else:
            new_state_value=self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_value)
            target_value=reward + gamma + max_new_state_values
episodes=100
solved=0
steps_ep=[]
total_episodes=[]
qnet_agent=QNet_Agent()
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    gamma= 1
    learning_rate=1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,4)
    print(state_n)
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        print(stat[0],state_n[0])
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
print('The simulation has been solved the environment Neural Network:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.title('The simulation has been solved the environment Neural Network:{}'.format(solved/episodes))
plt.show()

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import random 
import math 
import time
from torch.autograd import Variable

#if gpu is to be used 
use_cuda=torch.cuda.is_available()

device=torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

learning_rate=0.01
number_of_inputs = 4
number_of_outputs = 4
egreedy=0.5
update_target_frequency = 500
batch_size=32
hidden_layer=24
steps=[]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,number_of_outputs)
        self.linear2 = nn.Linear(hidden_layer,number_of_outputs)
        self.activation = nn.Tanh()
    def forward(self, x):
        output = self.linear1(x)
        output = self.activation(output)
        output1 = self.linear2(output)
        return output1

class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss() 
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        self.update_target_counter = 0
    def select_action(self,state):
         random_for_egreedy = torch.rand(1)[0]
         if random_for_egreedy > egreedy:
             with torch.no_grad():
                 state=Tensor(state).to(device)
                 action_from_nn=self.nn(state)
                 action=torch.max(action_from_nn,0)[1]
                 action=action.item()
         else:
            random_values=Q[int(state_n[1][0])] + np.random.randint(2, size=(1,max_moves))/1000
            actiona=np.argmax(random_values)
            random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
            actionb=np.argmax(random_values)
            action=(actiona,actionb)
         return action       
    def optimize(self,state,action,new_state,reward,done):
        state=Tensor(state).to(device)
        new_state=Tensor(new_state).to(device)
        reward = Tensor(reward+1).to(device)
        action = LongTensor(action).to(device)
        #done = Tensor(done).to(device)
        if done:
            target_value = reward
        else:
            new_state_value=self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_value, 1)[0]
            target_value=reward + (1- done) * gamma * max_new_state_values
            predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
            
            loss = self.loss_func(predicted_value, target_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.update_target_counter % update_target_frequency == 0:
                self.target_nn.load_state_dict(self.nn.state_dict())
            
            self.update_target_counter +=1
            
            
episodes=100
solved=0
steps_ep=[]
total_episodes=[]
qnet_agent=QNet_Agent()
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    gamma= 1
    learning_rate=1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,4)
    print(state_n)
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        print('This is the action {}'.format(action))
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        print(stat[0],state_n[0])
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        qnet_agent.optimize(stat[0],action,state_n[0],re,do)
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        #print(Q)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment DQN:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.show()


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

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()



def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    #test_probs = probs[0].detach().numpy()
    #nprobs = np.nan_to_num(test_probs, copy=True, nan=0.0, posinf=None, neginf=None)
    #print('This tensor is nan {}'.format(torch.any(probs.isnan())))
    #if torch.any(probs.isnan()) == True:
    print(probs)
    probs=torch.nan_to_num(probs)
    probs=probs+0.01
        #probs = torch.tensor([0., 0., 0., 0.], requires_grad=True)
        #probs != probs
    #print('Those are the probs {}'.format(probs))
    #nprobs = torch.from_numpy(nprobs, requires_grad=True).float().unsqueeze(0)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def mainly():
    import matplotlib.pyplot as plt
    running_reward = 100
    episodes=100
    solved=0
    steps_ep=[]
    total_episodes=[]
    for i_episode in range(0,episodes):
        print('This is the episode number {}'.format(i_episode))
        state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        #state, _ = env.reset()
        ep_reward = 0
        do=False
        reward_episode=[]
        while do!=True:  # Don't infinite loop while learning
            print(state_n)
            actiona = select_action(state_n[0])
            actionb = select_action(state_n[0])
            action = (actiona,actionb)
            print('This is the action {}'.format(action))
            #state, reward, done, _, _ = env.step(action)
            stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
            print('This is the reward {}'.format(re))
            reward_episode.append(re)
            if args.render:
                #env.render()
                print('render episode')
            policy.rewards.append(re)
            ep_reward += re
            state_n=stat
            #steps_ep.append(len(reward_episode))
            #if ep_reward > 0:
            #    solved+=1
            if reward_episode[-1]==1:
                solved+=1
                steps_ep.append(len(reward_episode))
            #policy.rewards.append(re)
            #ep_reward += re
            total_episodes.append(reward_episode[-1])

            #if done:
            #    break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 100:
            #print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
             #     i_episode, ep_reward, running_reward))
            print('Average reward: {:.2f}'.format(
                   running_reward))
            break
        else:
            continue
        #if running_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(running_reward, t))
    print('The simulation has been solved the environment PPO:{}'.format(solved/episodes))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
    plt.plot(total_episodes)
    plt.title('The simulation has been solved the environment PPO:{}'.format(solved/episodes))
    plt.show()
    return total_episodes

if __name__ == '__main__':
    tot_ep=mainly()


#import torch
#import torch.nn as nn
import numpy as np
#import torch.optim as optim
import gym
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
HISTORY_LENGTH = 1  
#hyperparameters
D = 4#len(env.reset())*HISTORY_LENGTH
M = 20
K = 4  
#print('This is the D {} M {} and K {}'.format(D,M,K))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

   
def relu(x):
   return x *(x >0)
    
class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D=D
        self.M=M
        self.K=K
        self.f=f
    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) /np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K)/ np.sqrt(M)
        self.b2 = np.zeros(K)
            
    def forward(self, X):
        #self.f
        #print('This is the input data X{}'.format(X[0]))
        #print('This is the w1 {}'.format(self.W2))
        #print('This is the b1 {}'.format(self.b2.shape))
        Z = np.tanh(X.dot(self.W1)+ self.b1)
        return softmax(Z.dot(self.W2)+ self.b2)
        
    def sample_action(self, x):
        #X=np.atleast_2d(x)
        X=x
        Y=self.forward(X)
        #print('Forward process {}'.format(P))
        y=Y[0]
        return np.argmax(y)
        
    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
        
    def get_params_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
            }
    def set_params(self, params):
        D,M,K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D*M:D*M +M]
        self.W2 = params[D * M + M:D*M+M+M*K].reshape(M,K)
        self.b2 = params[-K:]
        
def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters):
    #assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    learning_rate = np.zeros(num_iters)
    sigma_v = np.zeros(num_iters)
    parms = np.zeros(num_iters)
    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        ### slow way
        R = np.zeros(population_size)
        acts=np.zeros(population_size)
        #loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j],acts[j] = f(params_try)
        m = R.mean()
        s = R.std()+0.001
        #print('This is s {}'.format(s))
        if s == 0:
            # we cannot apply the following equation
            print("Skipping")
            continue
        
        A = (R-m)/s
        reward_per_iteration[t]= m
        params = params + lr/(population_size*sigma)+np.dot(N.T, A)
        parms[t]=params.mean() 
        learning_rate[t]=lr
        sigma_v[t]=sigma
        #update the learning rate
        #lr *= 0.001
        lr *=0.992354
        sigma += 0.7891
        print("Iter:",t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:",(datetime.now()-t0))
        if m > R.max()/1.5 or R.max() > 300:
            actis = acts
        else:
            actis=np.zeros(population_size)
    return params, reward_per_iteration,actis,learning_rate,sigma_v,population_size,parms
    
def reward_function(params):
    model = ANN(D, M, K)
    model.set_params(params)
    # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    obs = state_n[0]#obs[0]
    obs_dim= len(obs)
    if HISTORY_LENGTH >1:
        state =np.zeros(HISTORY_LENGTH*obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
        #get the action
        #print('This is the state {}'.format(state))
        actiona = model.sample_action(state)
        actionb = model.sample_action(state)
        action=(actiona,actionb)
        #print('This is the action {}'.format(action))
        #print('This is the action {}'.format(action))
        #perform the action
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #update total reward
        obs=stat[0]
        done=do
        episode_reward += re
        episode_length +=1
        #update state
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,int(''.join(map(str,action)))
    
if __name__=='__main__':
    model = ANN(D,M,K)
    if len(sys.argv) > 1 and sys.argv[1] =='play':
        #play with a saved model
        j = np.load('es_cartpole_results.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        # in case intial shapes are not correct
        D, M =j['W1'].shape
        K = len(j['b2'])
        model.D, model.M, model.K = D, M, K
        x=np.arange(0,len(j['train']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['train'], label='Rewards')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Best reward={j['train'].max()}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Rewards')
        plt.savefig("Rewards-evolutionstrategy.png")
        plt.show()
        x=np.arange(0,len(j['learning_rate_v']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['learning_rate_v'], label='learning_rate')
        plt.plot(x, j['sigmav'], label='sigma')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Population size={j['populat_s']}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Learing rate')
        plt.savefig("learning_rate-evolutionstrategy.png")
        plt.show()
        x=np.arange(0,len(j['learning_rate_v']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['pmeters'], label='weights')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Weight Average={j['pmeters'].mean()}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Weights')
        plt.savefig("NN-evolutionstrategy.png")
        plt.show()
    else:
        # train and save
        model.init()
        params = model.get_params()
        best_params, rewards, actions, learn_r,sigmv,pop_s,parms = evolution_strategy(
            f=reward_function,
            population_size=80,
            sigma=0.1,
            lr=0.0003,
            initial_params=params,
            num_iters=450,
        )
            
        model.set_params(best_params)
        np.savez('es_cartpole_results.npz',
                 learning_rate_v=learn_r,
                 sigmav=sigmv,
                 populat_s=pop_s,
                 pmeters=parms,
                 actions_e=actions,
                 train=rewards,
                 **model.get_params_dict(),
        )
        #play 5 test episodes
        #env.set_display(True)
        state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        #env.reset()
        for t in range(0,len(actions)):
            render()
            stat,re,do,action_h,bob_key=step(actions[t],act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        for _ in range(5):
            print("Test:", reward_function(best_params))