#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:46:09 2022

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
Rewards accumulate: negative points for wrong guess, positive points for correct guess
Game terminates with correct key or N moves
"""
class Qprotocol:
    def __init__(self,maxm):
        self.max_moves = maxm
        # State for alice
        self.data1 = np.random.randint(0,2,1)
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
                # reward = 0
            if( len(self.bob_key) == len(self.data1) ):
            # self.done = True
                if( np.array(self.bob_key).all() == self.data1.all() ):
                    reward = +1
                    self.cumulative_reward += reward
                    self.done = True
                else:
                    reward = -1
                    self.cumulative_reward += reward
                    # Add 1 to key - Bob should decide to take this action based on his datalog
        if( action_bob == 3 ):
            self.bob_key.append(1)
            # reward = 0
            # If bob wrote enough bits
            if( len(self.bob_key) == len(self.data1) ):
                # self.done = True
                if( np.array(self.bob_key).all() == self.data1.all() ):
                    reward = +1
                    self.cumulative_reward += reward
                    self.done = True
            else:
                reward = -1
        self.cumulative_reward += reward
        # Update the actions that alice and bob took
        self.alice_observation[(len(self.action_history)-1)%len(self.alice_observation)] = action[0]
        self.bob_observation = np.concatenate(([self.bob_has_mail], self.bob_datalog))
        state = (self.alice_observation, self.bob_observation)
        return state, reward, self.done, {'action_history':self.action_history}
    def reset(self):
        import numpy as np
        self.max_moves = 4
        # State for alice
        self.data1 = np.random.randint(0,2,1)
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
        return state
    def render(self):
        print("---Alice---")
        print("- Datalog: ", self.alice_datalog)
        print("---Bob---")
        print("- Has Mail: ", self.bob_has_mail)
        print("- Datalog: ", self.bob_datalog)
        print("")
        return
# =============================================================================
# Q-learning deterministic
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt	
episodes=100
qp=Qprotocol(4)
solved=0
steps_ep=[]
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]   
q=(6,len(actions_list))
Q=np.zeros(q)
total_episodes=[]
q_value=[]
for episode in range(episodes):
    #np.random.seed(0)
    gamma= 1
    state=qp.reset()
    #print(state_n)
    #state_n=state
    done=False
    reward_episode=[]
    steps=0
    #for t in range(0,15):
    while done!=True:
        steps+=1
        random_values=Q[int(state[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        print(actiona,np.argmax(state[0][0]))
        q_val=(int(np.argmax(state[0])),actiona)
        print('Print q_val {}'.format(q_val))
        #random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
        #actionb=np.argmax(random_values)
        action=np.array(actions_list[actiona])#(actiona,actionb)
        #print('This is the action {}'.format(action))
        new_state, reward, done,info=qp.step(action)
        #stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        value=reward + gamma * max(Q[int(new_state[0][0])])
        q_value.append(value)
        #print('This is the Q-table 2 {}'.format(q_val))
        Q[(q_val)]=value
        #print(re)
        #print('This is the reward {}'.format(reward))
        reward_episode.append(reward)
        #print(Q)
        state=new_state
    if reward_episode[-1]==1:
        solved+=1 
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment Belman Equation:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Bellman Equation:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(q_value,c='orange')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The q-value during the training')
plt.show()
total_re=[]
solved=0
r=0
cumulative_reward=[]
episodes = 100
for _ in range(episodes):
    state=qp.reset()
    print(state[0][0])
#   epochs, penalties, reward = 0, 0, 0
    state=state[0][0]
    done = False
    while not done:
        action = np.argmax(Q[int(state)])
        #print(action)
        action=np.array(actions_list[action])
        new_state, reward, done,info=qp.step(action)
        state=new_state[0][0]
        #print(re)
        if done==True:
            total_re.append(reward)
            if reward==1:
                solved+=1
    r+=total_re[-1]
    cumulative_reward.append(r)

plt.figure(figsize=(13, 13))            
print('The simulation has been solved the environment Belman Equation:{}'.format(solved))
plt.plot(total_re)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Bellman Equation:{}'.format(solved/episodes))
plt.show()
plt.figure(figsize=(13, 13))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
#plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()


# =============================================================================
#  Q-learning Stochastic 
# =============================================================================


episodes=100
solved=0
#actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q=np.zeros(q)
total_episodes=[]
q_value_ep=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state=qp.reset()
    done=False
    reward_episode=[]
    steps=0
    while done!=True:
        steps+=1
        random_values=Q[int(state[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state[0][0]),actiona)
        action=np.array(actions_list[actiona])
        new_state, reward, done,info=qp.step(action)
        print('this is the reward {}'.format(reward))
        q_value=(1-learning_rate)*Q[q_val]+learning_rate * (reward + gamma * max(Q[int(new_state[0][0])]))
        q_value_ep.append(q_value)
        Q[q_val]=q_value
        reward_episode.append(reward)
        state=new_state
    print('This is reward episode {}'.format(reward_episode))
    if reward_episode[-1]==1:
        solved+=1
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning stochastic:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Q learning stochastic:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(q_value_ep,c='orange')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The q-value during the training')
plt.show()

total_re=[]
solved=0
r=0
cumulative_reward=[]
episodes = 100
for _ in range(episodes):
    state=qp.reset()
    print(state[0][0])
#   epochs, penalties, reward = 0, 0, 0
    state=state[0][0]
    done = False
    while not done:
        action = np.argmax(Q[int(state)])
        #print(action)
        action=np.array(actions_list[action])
        new_state, reward, done,info=qp.step(action)
        state=new_state[0][0]
        #print(re)
        if done==True:
            total_re.append(reward)
            if reward==1:
                solved+=1
    r+=total_re[-1]
    cumulative_reward.append(r)
plt.figure(figsize=(13, 13))            
print('The simulation has been solved the environment Q-learning Equation stochastic:{}'.format(solved))
plt.plot(total_re)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Q-learning Equation stochastic:{}'.format(solved/episodes))
plt.show()
plt.figure(figsize=(13, 13))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
#plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random 
import math
import time
import matplotlib.pyplot as plt 
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
#seed_value=0
#torch.manual_seed(seed_value)
#random.seed(seed_value)
learning_rate=0.01
#num_episodes=500
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
                 #print('This is the action {} in egreedy'.format(action))
         else:
            print('This is the state {}'.format(state))
            random_values=Q[int(state[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            actiona=np.argmax(random_values)
            #random_values=Q[int(abs(state_n[1][1]))] + np.random.randint(2, size=(1,max_moves))/1000
            #actionb=np.argmax(random_values)
            action=np.array(actions_list[actiona])
         return action       
    def optimize(self,state,action,new_state,reward,done):
        state=Tensor(state).to(device)
        new_state=Tensor(new_state).to(device)
        #reward = Tensor(reward).to(device)
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
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    learning_rate=1
    state=qp.reset()
    q=(4,len(actions_list))
    print(state)
    Q=np.zeros(q)
    done=False
    reward_episode=[]
    steps=0
    while done!=True:
        steps+=1
        frames_total+=1
        epsilon=calculate_epsilon(frames_total)
        action=qnet_agent.select_action(state[0],epsilon)
        print('This is the action {}'.format(action))
        new_state, reward, done,info=qp.step(action)
        qnet_agent.optimize(new_state[0],action,state[0],reward,done)
        reward_episode.append(reward)
        state=new_state
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

episodes=100
solved=0
frames_total=0
steps_ep=[]
total_episodes=[]
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    gamma= 1
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    learning_rate=1
    state=qp.reset()
    done=False
    reward_episode=[]
    steps=0
    while done!=True:
        steps+=1
        frames_total+=1
        epsilon=calculate_epsilon(frames_total)
        action=qnet_agent.select_action(state[0],epsilon)
        new_state, reward, done,info=qp.step(action)
        reward_episode.append(reward)
        state=new_state
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
qp=Qprotocol(4)
# #hyperparameters
D = 4#len(env.reset())*HISTORY_LENGTH
M = 32
K = 12


actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]


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
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) /np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K)/ np.sqrt(M)
        self.b2 = np.zeros(K)
            
    def forward(self, X):
        Z = np.tanh(np.dot(X,self.W1)+ self.b1)
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
            R[j],acts[j],_ = f(params_try)
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
    counterr=0
    state=qp.reset()
    obs = state[0]
    obs_dim= len(obs)
    if HISTORY_LENGTH >1:
        state =np.zeros(HISTORY_LENGTH*obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
        actiona = model.sample_action(state)
        action=np.array(actions_list[actiona])
#         #perform the action
        new_state, reward, done,info=qp.step(action)
#         #update total reward
        #done=do
        obs=new_state[0]
        episode_reward += reward
        episode_length +=1
#         #update state
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona,episode_length
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
            population_size=500,
            sigma=0.002,
            lr=0.006,
            initial_params=params,
            num_iters=100,
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
        total_episodes=[]
        solved=0
        episodes=100
        Rewa=0
        cum_re=[]
        steps_ep=[]
        for _ in range(episodes):
            Rew, ac,steps=reward_function(best_params)
            Rewa+=Rew
            total_episodes.append(Rew)
            Rewa += total_episodes[-1]
            cum_re.append(Rewa)
            #print(ac)
            steps_ep.append(steps)
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
    print('Average steps per episode {}'.format(np.mean(steps_ep)))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

# Sample action from actor
#modela=_build_model()
@tf.function
def sample_action(observation):
    logits = actor(observation)
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
clip_ratio = 0.001
policy_learning_rate = 0.25#3e-4
value_function_learning_rate = 0.25#1e-3
train_policy_iterations = 5000
train_value_iterations = 5000
lam = 0.97
target_kl = 0.01
hidden_sizes = (32, 32)

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

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rewards_during_training=[]
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
episode_return=0
episode_length=0
# Iterate over the number of epochs
q_value_critic=[]
action_actor=[]
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    done=False
    state=qp.reset()
    observation=np.array(state[0])
    while done != True:
    # Iterate over the steps of each epoch
    #for t in range(steps_per_epoch):
        if render:
            print(render)
            #env.render()
        observation = observation.reshape(1, -1)
        #print('Observation shape {}'.format(observation))
        #observation=observation[0]
        logits, actiona = sample_action(observation)
        log=np.array(logits[0])
        action_actor.append(log[actiona])
        actiona=actiona[0]
        #print(actiona)
        #print(np.argmax(logits[0]))
        #actiona = np.argmax(logits[0])
        #print('This are logits {} and actions {}'.format(logits, actiona))
        action=np.array(actions_list[actiona])
        #print('This is the action {}'.format(action))
        new_state, reward, done,info=qp.step(action)
        #observation_new, reward, done, _,op = env.step(action[0].numpy())
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
            #print(episode_return)
            sum_length += episode_length
            num_episodes += 1
            state=qp.reset()
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
#def save_weights():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    actor.save(path+ '_actor.h5')
#    critic.save(path+ '_critic.h5')
#def load_weights():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
#    critic.load_weights(path+ '_critic.h5')
#    actor.load_weights(path+ '_actor.h5')
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
    state=qp.reset()
    ep_reward = 0
    done=False
    observation=state[0]
    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    while done!=True:
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
        actiona=actiona[0]
        #print('This are logits {} and actions {}'.format(logits, actiona))
        action=np.array(actions_list[actiona])
        #print('This is the action {}'.format(action))
        new_state, reward, done,info=qp.step(action)
        #observation_new, reward, done, _,op = env.step(action[0].numpy())
        print('This is the reward {}'.format(reward))
        episode_return += reward
        episode_length += 1
        observation=new_state[0]
        if done==True:
            total_episodes.append(reward)
        if reward==1:
            solved+=1
    

plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation:{}'.format(solved/episodes))
plt.show()
