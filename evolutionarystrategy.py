#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:54:12 2022

@author: kel
"""
#import torch
#import torch.nn as nn
import numpy as np
#import torch.optim as optim
import gym
import random
import math
import time
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from ple import PLE
#from ple.games.flappybird import FlappyBird
import sys

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
	

# =============================================================================
# 
# HISTORY_LENGTH = 1
#   
# #hyperparameters
# D = 4 #len(env.reset())*HISTORY_LENGTH
# M = 20
# K = 2    
# print('This is the D {} M {} and K {}'.format(D,M,K))
# def softmax(a):
#    c = np.max(a, axis=1, keepdims=True)
#    e = np.exp(a-c)
#    return e/e.sum(axis=-1, keepdims=True)
# 
# 
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
# 
#    
# def relu(x):
#    return x *(x >0)
#     
# class ANN:
#     def __init__(self, D, M, K, f=relu):
#         self.D=D
#         self.M=M
#         self.K=K
#         self.f=f
#     def init(self):
#         D, M, K = self.D, self.M, self.K
#         self.W1 = np.random.randn(D, M) /np.sqrt(D)
#         self.b1 = np.zeros(M)
#         self.W2 = np.random.randn(M, K)/ np.sqrt(M)
#         self.b2 = np.zeros(K)
#             
#     def forward(self, X):
#         #self.f
#         #print('This is the input data X{}'.format(X[0]))
#         #print('This is the w1 {}'.format(self.W2))
#         #print('This is the b1 {}'.format(self.b2.shape))
#         Z = np.tanh(X.dot(self.W1)+ self.b1)
#         return softmax(Z.dot(self.W2)+ self.b2)
#         
#     def sample_action(self, x):
#         X=np.atleast_2d(x)
#         #print('This is X of np.atleast {}'.format(X))
#         #X=x
#         Y=self.forward(X)
#         #print('Forward process {}'.format(P))
#         y=Y[0]
#         return np.argmax(y)
#         
#     def get_params(self):
#         return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
#         
#     def get_params_dict(self):
#         return {
#             'W1': self.W1,
#             'b1': self.b1,
#             'W2': self.W2,
#             'b2': self.b2
#             }
#     def set_params(self, params):
#         D,M,K = self.D, self.M, self.K
#         self.W1 = params[:D * M].reshape(D, M)
#         self.b1 = params[D*M:D*M +M]
#         self.W2 = params[D * M + M:D*M+M+M*K].reshape(M,K)
#         self.b2 = params[-K:]
#         
# def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters):
#     #assume initial params is a 1-D array
#     num_params = len(initial_params)
#     print('This is the number of parameters {}'.format(num_params))
#     reward_per_iteration = np.zeros(num_iters)
#     print('This is the initialization rewards{} the shape of reward {}'.format(reward_per_iteration,reward_per_iteration.shape))
#     learning_rate = np.zeros(num_iters)
#     print('This is the initialization learning rate {} the shape of learning_rate {}'.format(learning_rate,learning_rate.shape))
#     sigma_v = np.zeros(num_iters)
#     print('This is the initialization sigma v {} the shape of v {}'.format(sigma_v,sigma_v.shape))
#     parms = np.zeros(num_iters)
#     print('This is the parameters initialization {} the shape of parms {}'.format(parms,len(parms)))
#     params = initial_params
#     for t in range(num_iters):
#         t0 = datetime.now()
#         N = np.random.randn(population_size, num_params)
#         print('This is the N with randon values populationsize number of parameters {} and the N size is {}'.format(N,N.shape))
#         ### slow way
#         R = np.zeros(population_size)
#         print('This is the R with zero {} and this the shape of R {}'.format(R,R.shape))
#         acts=np.zeros(population_size)
#         print('This is the actions initialization {} and this the shape of acts {}'.format(acts,acts.shape))
#         #loop through each "offspring"
#         for j in range(population_size):
#             print('for j in population size {} this is the N[j] {} this is the shape {}'.format(j,N[j],N[j].shape))
#             params_try = params + sigma * N[j]
#             print('This is the params try that is used each time to evalutate the action and reward {} and this is the shape {}'.format(params_try,len(params_try)))
#             R[j],acts[j] = f(params_try)
#             print('This is the Rj {} and this is the action j {}'.format(R[j],acts[j]))
#         m = R.mean()
#         print('This is the m {}'.format(m))
#         s = R.std()+0.001
#         print('This is the s {}'.format(s))
#         #print('This is s {}'.format(s))
#         if s == 0:
#             # we cannot apply the following equation
#             print("Skipping")
#             continue
#         
#         A = (R-m)/s
#         print('This is A {}'.format(A))
#         reward_per_iteration[t]= m
#         print('This is reward_per_iteration {}'.format(reward_per_iteration[t]))
#         params = params + lr/(population_size*sigma)+np.dot(N.T, A)
#         print('this the new params {}'.format(params))
#         parms[t]=params.mean() 
#         print('This is the parms[t] {}'.format(parms[t]))
#         learning_rate[t]=lr
#         sigma_v[t]=sigma
#         #update the learning rate
#         #lr *= 0.001
#         lr *=0.992354
#         sigma += 0.7891
#         print("Iter:",t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:",(datetime.now()-t0))
#         if m > R.max()/1.5 or R.max() > 300:
#             actis = acts
#         else:
#             actis=np.zeros(population_size)
#     return params, reward_per_iteration,actis,learning_rate,sigma_v,population_size,parms
#     
# def reward_function(params):
#     model = ANN(D, M, K)
#     model.set_params(params)
#     # play one episode and return the total reward
#     episode_reward = 0
#     episode_length = 0
#     done = False
#     obs = env.reset()
#     obs = obs[0]
#     obs_dim= len(obs)
#     if HISTORY_LENGTH >1:
#         state =np.zeros(HISTORY_LENGTH*obs_dim)
#         state[obs_dim:] = obs
#     else:
#         state = obs
#     while not done:
#         #get the action
#         #print('This is the state {}'.format(state))
#         action = model.sample_action(state)
#         #print('This is the action {}'.format(action))
#         #print('This is the action {}'.format(action))
#         #perform the action
#         obs, reward, done, val, _ = env.step(action)
#         #update total reward
#         episode_reward += reward
#         episode_length +=1
#         #update state
#         if HISTORY_LENGTH > 1:
#             state = np.roll(state, -obs_dim)
#             state[-obs_dim:]=obs
#         else:
#             state = obs
#     return episode_reward,action
#     
# if __name__=='__main__':
#     model = ANN(D,M,K)
#     if len(sys.argv) > 1 and sys.argv[1] =='play':
#         #play with a saved model
#         j = np.load('es_cartpole_results.npz')
#         best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
#         # in case intial shapes are not correct
#         D, M =j['W1'].shape
#         K = len(j['b2'])
#         model.D, model.M, model.K = D, M, K
#         x=np.arange(0,len(j['train']))
#         plt.figure(figsize=(13, 13))
#         plt.plot(x, j['train'], label='Rewards')
#         plt.grid(True,which="both",ls="--",c='gray') 
#         plt.title(f"Best reward={j['train'].max()}")
#         plt.legend()
#         plt.xlabel(f'Number of Steps of episode')
#         plt.ylabel('Rewards')
#         plt.savefig("Rewards-evolutionstrategy.png")
#         plt.show()
#         x=np.arange(0,len(j['learning_rate_v']))
#         plt.figure(figsize=(13, 13))
#         plt.plot(x, j['learning_rate_v'], label='learning_rate')
#         plt.plot(x, j['sigmav'], label='sigma')
#         plt.grid(True,which="both",ls="--",c='gray') 
#         plt.title(f"Population size={j['populat_s']}")
#         plt.legend()
#         plt.xlabel(f'Number of Steps of episode')
#         plt.ylabel('Learing rate')
#         plt.savefig("learning_rate-evolutionstrategy.png")
#         plt.show()
#         x=np.arange(0,len(j['learning_rate_v']))
#         plt.figure(figsize=(13, 13))
#         plt.plot(x, j['pmeters'], label='weights')
#         plt.grid(True,which="both",ls="--",c='gray') 
#         plt.title(f"Weight Average={j['pmeters'].mean()}")
#         plt.legend()
#         plt.xlabel(f'Number of Steps of episode')
#         plt.ylabel('Weights')
#         plt.savefig("NN-evolutionstrategy.png")
#         plt.show()
#     else:
#         # train and save
#         model.init()
#         params = model.get_params()
#         best_params, rewards, actions, learn_r,sigmv,pop_s,parms = evolution_strategy(
#             f=reward_function,
#             population_size=2,
#             sigma=0.5,
#             lr=0.10,
#             initial_params=params,
#             num_iters=1,
#         )
#             
#         model.set_params(best_params)
#         np.savez('es_cartpole_results.npz',
#                  learning_rate_v=learn_r,
#                  sigmav=sigmv,
#                  populat_s=pop_s,
#                  pmeters=parms,
#                  actions_e=actions,
#                  train=rewards,
#                  **model.get_params_dict(),
#         )
#         #play 5 test episodes
#         #env.set_display(True)
#         env.reset()
#         for t in range(0,len(actions)):
#             env.render()
#             env.step(int(actions[t]))
#         for _ in range(5):
#             print("Test:", reward_function(best_params))
# 
# =============================================================================

#import torch
#import torch.nn as nn
import numpy as np
#import torch.optim as optim
import gym
import random
import math
import time
# import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
HISTORY_LENGTH = 1  
# #hyperparameters
D = 4#len(env.reset())*HISTORY_LENGTH
M = 32
K = 12

np.random.seed(0)
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
#print('This is the D {} M {} and K {}'.format(D,M,K))
#print('This is the D {} M {} and K {}'.format(D,M,K))
#def softmax(a):
#   c = np.max(a, axis=1, keepdims=True)
#   e = np.exp(a-c)
#   return e/e.sum(axis=-1, keepdims=True)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

   
def relu(x):
   return x *(x >0)
    
class ANN:
    #def __init__(self, D, K, f=relu):
    def __init__(self, D, M, K, f=relu):
        self.D=D
        self.M=M
        self.K=K
        self.f=f
    def init(self):
        #D, K = self.D, self.K
        #self.W1 = np.random.randn(D, K) /np.sqrt(D)
        #self.b1 = np.zeros(K)        
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) /np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K)/ np.sqrt(M)
        self.b2 = np.zeros(K)
            
    def forward(self, X):
        #self.f
        #print('This is the input data X{}'.format(X))
        #print('This is the w1 {}'.format(self.W2))
        #print('This is the b1 {}'.format(self.b2.shape))
        Z = np.tanh(np.dot(X,self.W1)+ self.b1)
        #return softmax(Z) 
        return softmax(Z.dot(self.W2)+ self.b2)
        
    def sample_action(self, x):
        X=np.atleast_2d(x)
        #X=x
        Y=self.forward(X)
        #print('Forward process {}'.format(Y))
        y=Y[0]
        return np.argmax(y)
        
    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
        
    def get_params_dict(self):
        #return {
         #   'W1': self.W1,
         #   'b1': self.b1
          #  'W2': self.W2,
          #  'b2': self.b2
        #    }
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
            }
    def set_params(self, params):
        #D,K = self.D, self.K
        #self.W1 = params[:D * K].reshape(D, K)
        #self.b1 = params[D*K:D*K +K]
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
        print('This is the number of acts {}'.format(len(acts)))
        #loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j],acts[j] = f(params_try)
            #print('This is the action {}'.format(acts[j]))
            #print('This is the reward {}'.format(R[j]))
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
        if m > 0.01:# or R.max() >= 1:#m > R.max()/1.5 or R.max() >= 1:
            actis = acts
            print('True')
            #break
        else:
            actis=np.zeros(population_size)
    return params, reward_per_iteration,actis,learning_rate,sigma_v,population_size,parms
    
def reward_function(params):
    #model=ANN(D,K)
    model = ANN(D, M, K)
    #models = ANN(D, M, K)
    model.set_params(params)
    #models.set_params(params)
#     # play one episode and return the total reward
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
#         #get the action
#         #state=np.array(state)
        #print('This is the state {}'.format(state))
        actiona = model.sample_action(state)
        action=np.array(actions_list[actiona])
#        actionb = model.sample_action(state)
   #     action=np.array(actions_list[action])
        #action=(actiona,actionb)
        #print('This is the action {}'.format(action))
        #print('This is the action {}'.format(action))
#         #perform the action
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
#         #update total reward
        done=do
        obs=stat[0]
# #        print(obs)
        episode_reward += re
        episode_length +=1
#         #update state
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona
#     
if __name__=='__main__':
    #model=ANN(D,K)
    model = ANN(D,M,K)
    if len(sys.argv) > 1 and sys.argv[1] =='play':
        #play with a saved model
        j = np.load('es_qkprotocol_results.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        # in case intial shapes are not correct
        D, M =j['W1'].shape
        K = len(j['b2'])
        model.D, model.M, model.K = D, M, K
        #model.D,model.K=D,K
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
            population_size=50,
            sigma=0.28,
            lr=0.36,
            initial_params=params,
            num_iters=4600,
        )
            
        model.set_params(best_params)
        np.savez('es_qkprotocol_results02.npz',
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
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        #env.reset()
        #for t in range(0,len(actions)):
        #    render()
        #    stat,re,do,action_h,bob_key=step(actions[t],act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        total_episodes=[]
        solved=0
        episodes=100
        Rewa=0
        cum_re=[]
        for _ in range(episodes):
            Rew, ac=reward_function(best_params)
            Rewa+=Rew
            total_episodes.append(Rew)
            Rewa += total_episodes[-1]
            cum_re.append(Rewa)
            if Rew>0:
                solved+=1
            print("Episode {} Reward per episode {}".format(_,Rew))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    #x=np.arange(0,len(j['learning_rate_v']))
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{}'.format(solved/episodes))
    plt.show()
    #plt.grid(True,which="both",ls="--",c='gray') 
    #plt.title(f"Weight Average={j['pmeters'].mean()}")
    #plt.legend()
    #plt.xlabel(f'Number of Steps of episode')
    #plt.ylabel('Rewards')
    #plt.savefig("NN-evolutionstrategy.png")
    #plt.show()
    #print('The simulation has been solved the environment Evolutionary Strategy:{} learning rate {} sigma {}'.format(solved/episodes,learn_r[-1],sigmv[-1]))
    #plt.plot(total_episodes)
    #plt.title('The simulation has been solved the environment Evolutionary Strategy:{}'.format(solved/episodes))
    #plt.show()
