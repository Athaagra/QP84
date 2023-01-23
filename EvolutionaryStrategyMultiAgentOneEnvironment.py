#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:35:33 2022
@author: Optimus
"""

import numpy as np
import matplotlib.pyplot as plt
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
#import random
#random.seed(0)
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

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
HISTORY_LENGTH = 0  
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
        Y=self.forward(X)
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
        
def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters,inputme):
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
            R[j],acts[j],_,_ = f(params_try,inputme)
        m = R.mean()
        s = R.std()+0.001
        if s == 0:
            print("Skipping")
            continue
        
        A = (R-m)/s
        reward_per_iteration[t]= m
        params = params + lr/(population_size*sigma)+np.dot(N.T, A)
        parms[t]=params.mean() 
        learning_rate[t]=lr
        sigma_v[t]=sigma
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
    
def reward_function(params,inp):
    model = ANN(D, M, K)
    env=Qprotocol(4)
    inpu=inp
    model.set_params(params)
    episode_reward = 0
    episode_length = 0
    done = False
    counterr=0
    state_n=env.reset(4,inpu)
    obs = state_n[0]
    obs_dim= len(obs)
    if HISTORY_LENGTH >1:
        state =np.zeros(HISTORY_LENGTH*obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
        actiona = model.sample_action(state)
        action=np.array(actions_list[actiona])
        new_state, reward, done,info ,bob_key=env.step(action)
        obs=new_state[0]
        episode_reward += reward
        episode_length +=1
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona,episode_length,bob_key
#
def evol_strategy(mes):     
    model = ANN(D,M,K)
    model.init()
    params = model.get_params()
    message=mes
    best_params, rewards, actions, learn_r,sigmv,pop_s,parms = evolution_strategy(
                f=reward_function,
                population_size=500,
                sigma=0.002,
                lr=0.006,
                initial_params=params,
                num_iters=400,
                inputme=message)
    np.savez('es_qkprotocol_results'+str(message)+'.npz',
             learning_rate_v=learn_r,
             sigmav=sigmv,
             populat_s=pop_s,
             pmeters=parms,
             actions_e=actions,
             train=rewards,
             **model.get_params_dict(),)
    return best_params

def simulationev(inp,bp):
    total_fidelity=[]
    solved=0
    episodes=100
    Rewa=0
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        inputm=inp
        Rew, ac,steps,bobk=reward_function(bp,inputm)
        if Rew==1:
            bk=bobk
            steps_ep.append(steps)
        if Rew>0:
            if len(inp)==1:
                tp=LogicalStates[:,inputm].T*LogicalStates[bk,:]
                tp=tp[0]
                Fidelity=abs(sum(tp))**2
                steps_epi.append(steps_ep)
                total_fidelity.append(Fidelity)
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re.append(cumre)
            if len(inp)==2:
                inpus=''.join(str(x) for x in inp)
                bob_keys=''.join(str(x) for x in bk[:len(inp)])
                tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                Fidelity=abs(sum(tp))**2
                steps_epi.append(steps_ep)
                total_fidelity.append(Fidelity)
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re.append(cumre)
            if len(inp)==3:
                inpus=''.join(str(x) for x in inp)
                bob_keys=''.join(str(x) for x in bk[:len(inp)])
                tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                Fidelity=abs(sum(tp))**2
                steps_epi.append(steps_ep)
                total_fidelity.append(Fidelity)
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re.append(cumre)
            if len(inp)==4:
                inpus=''.join(str(x) for x in inp)
                bob_keys=''.join(str(x) for x in bk[:len(inp)])
                tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                Fidelity=abs(sum(tp))**2
                steps_epi.append(steps_ep)
                total_fidelity.append(Fidelity)
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re.append(cumre)
        else:
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(min(steps_ep),len(x))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {} '+str(len(inp))+''.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} '+str(len(inp))+''.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} '+str(len(inp))+''.format(sum(total_fidelity)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()



def onebitsimulation(inp,bp,bp1):
    total_fidelity=[]
    solved=0
    episodes=100
    Rewa=0
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        inputm=inp
        Rew, ac,steps,bobk=reward_function(bp,inputm)
        Rew1, ac1,steps1,bobk1=reward_function(bp1,inputm)
        if Rew==1:
            bk=bobk
            steps_ep.append(steps)
        if Rew1==1:
            bk=bobk1
            steps_ep.append(steps1)
        print(Rew,Rew1)
        if Rew>0 or Rew1>0:# or Rew2>0 or Rew3>0 or Rew4>0 or Rew5>0 or Rew6>0 or Rew7>0 or Rew8>0 or Rew9>0 or Rew10>0 or Rew11>0 or Rew12>0 or Rew13>0 or Rew15>0:
            tp=LogicalStates[:,inputm].T*LogicalStates[bk,:]
            tp=tp[0]
            Fidelity=abs(sum(tp))**2
            steps_epi.append(steps_ep)
            total_fidelity.append(Fidelity)
            solved+=1
            cumre+=1
            total_ep.append(1)
            cum_re.append(cumre)
        else:
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(min(steps_ep),len(x))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {} '+str(len(inp))+''.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} '+str(len(inp))+''.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} '+str(len(inp))+''.format(sum(total_fidelity)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()


def twobitsimulation(inp,bp,bp1,bp2,bp3):
    total_fidelity=[]
    solved=0
    episodes=100
    Rewa=0
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        inputm=inp
        Rew, ac,steps,bobk=reward_function(bp,inputm)
        Rew1, ac1,steps1,bobk1=reward_function(bp1,inputm)
        Rew2, ac2,steps2,bobk2=reward_function(bp2,inputm)
        Rew3, ac3,steps3,bobk3=reward_function(bp3,inputm)
        if Rew==1:
            bk=bobk
            steps_ep.append(steps)
        if Rew1==1:
            bk=bobk1
            steps_ep.append(steps1)
        if Rew2==1:
            bk=bobk2
            steps_ep.append(steps2)
        if Rew3==1:
            bk=bobk3
            steps_ep.append(steps3)
        print(Rew,Rew1)
        if Rew>0 or Rew1>0 or Rew2>0 or Rew3>0:# or Rew2>0 or Rew3>0 or Rew4>0 or Rew5>0 or Rew6>0 or Rew7>0 or Rew8>0 or Rew9>0 or Rew10>0 or Rew11>0 or Rew12>0 or Rew13>0 or Rew15>0:
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            steps_epi.append(steps_ep)
            total_fidelity.append(Fidelity)
            solved+=1
            cumre+=1
            total_ep.append(1)
            cum_re.append(cumre)
        else:
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(min(steps_ep),len(x))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {}'+str(len(inp))+''.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} '+str(len(inp))+''.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} '+str(len(inp))+''.format(sum(total_fidelity)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()

def threebitsimulation(inp,bp,bp1,bp2,bp3,bp4,bp5,bp6,bp7):
    total_fidelity=[]
    solved=0
    episodes=100
    Rewa=0
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        inputm=inp
        Rew, ac,steps,bobk=reward_function(bp,inputm)
        Rew1, ac1,steps1,bobk1=reward_function(bp1,inputm)
        Rew2, ac2,steps2,bobk2=reward_function(bp2,inputm)
        Rew3, ac3,steps3,bobk3=reward_function(bp3,inputm)
        Rew4, ac4,steps4,bobk4=reward_function(bp4,inputm)
        Rew5, ac5,steps5,bobk5=reward_function(bp5,inputm)
        Rew6, ac6,steps6,bobk6=reward_function(bp6,inputm)
        Rew7, ac7,steps7,bobk7=reward_function(bp7,inputm)
        if Rew==1:
            bk=bobk
            steps_ep.append(steps)
        if Rew1==1:
            bk=bobk1
            steps_ep.append(steps1)
        if Rew2==1:
            bk=bobk2
            steps_ep.append(steps2)
        if Rew3==1:
            bk=bobk3
            steps_ep.append(steps3)
        if Rew4==1:
            bk=bobk4
            steps_ep.append(steps4)
        if Rew5==1:
            bk=bobk5
            steps_ep.append(steps5)
        if Rew6==1:
            bk=bobk6
            steps_ep.append(steps6)
        if Rew7==1:
            bk=bobk7
            steps_ep.append(steps7)
        if Rew>0 or Rew1>0 or Rew2>0 or Rew3>0 or Rew4>0 or Rew5>0 or Rew6>0 or Rew7>0:
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            steps_epi.append(steps_ep)
            total_fidelity.append(Fidelity)
            solved+=1
            cumre+=1
            total_ep.append(1)
            cum_re.append(cumre)
        else:
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(steps_epi,len(x))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {}'+str(len(inp))+''.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} '+str(len(inp))+''.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} '+str(len(inp))+''.format(sum(total_fidelity)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()

def fourbitsimulation(inp,bp,bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10,bp11,bp12,bp13,bp14,bp15):
    total_fidelity=[]
    solved=0
    episodes=100
#    Rewa=0
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        inputm=inp
        Rew, ac,steps,bobk=reward_function(bp,inputm)
        Rew1, ac1,steps1,bobk1=reward_function(bp1,inputm)
        Rew2, ac2,steps2,bobk2=reward_function(bp2,inputm)
        Rew3, ac3,steps3,bobk3=reward_function(bp3,inputm)
        Rew4, ac4,steps4,bobk4=reward_function(bp4,inputm)
        Rew5, ac5,steps5,bobk5=reward_function(bp5,inputm)
        Rew6, ac6,steps6,bobk6=reward_function(bp6,inputm)
        Rew7, ac7,steps7,bobk7=reward_function(bp7,inputm)
        Rew8, ac8,steps8,bobk8=reward_function(bp8,inputm)
        Rew9, ac9,steps9,bobk9=reward_function(bp9,inputm)
        Rew10,ac10,steps10,bobk10=reward_function(bp10,inputm)
        Rew11, ac11,steps11,bobk11=reward_function(bp11,inputm)
        Rew12, ac12,steps12,bobk12=reward_function(bp12,inputm)
        Rew13, ac13,steps13,bobk13=reward_function(bp13,inputm)
        Rew14, ac14,steps14,bobk14=reward_function(bp14,inputm)
        Rew15, ac15,steps15,bobk15=reward_function(bp15,inputm)
        if Rew==1:
            bk=bobk
            steps_ep.append(steps)
        if Rew1==1:
            bk=bobk1
            steps_ep.append(steps1)
        if Rew2==1:
            bk=bobk2
            steps_ep.append(steps2)
        if Rew3==1:
            bk=bobk3
            steps_ep.append(steps3)
        if Rew4==1:
            bk=bobk4
            steps_ep.append(steps4)
        if Rew5==1:
            bk=bobk5
            steps_ep.append(steps5)
        if Rew6==1:
            bk=bobk6
            steps_ep.append(steps6)
        if Rew7==1:
            bk=bobk7
            steps_ep.append(steps7)
        if Rew8==1:
            bk=bobk8
            steps_ep.append(steps8)
        if Rew9==1:
            bk=bobk9
            steps_ep.append(steps9)
        if Rew10==1:
            bk=bobk10
            steps_ep.append(steps10)
        if Rew11==1:
            bk=bobk11
            steps_ep.append(steps11)
        if Rew12==1:
            bk=bobk12
            steps_ep.append(steps12)
        if Rew13==1:
            bk=bobk13
            steps_ep.append(steps13)
        if Rew14==1:
            bk=bobk14
            steps_ep.append(steps14)
        if Rew15==1:
            bk=bobk15
            steps_ep.append(steps15)
        if Rew>0 or Rew1>0 or Rew2>0 or Rew3>0 or Rew4>0 or Rew5>0 or Rew6>0 or Rew7>0 or Rew8>0 or Rew9>0 or Rew10>0 or Rew11>0 or Rew12>0 or Rew13>0 or Rew14>0 or Rew15>0:
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            steps_epi.append(steps_ep)
            total_fidelity.append(Fidelity)
            solved+=1
            cumre+=1
            total_ep.append(1)
            cum_re.append(cumre)
        else:
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode '+str(len(inp))+'')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_epi))
    steps=np.repeat(steps_epi,len(x))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {} '+str(len(inp))+''.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} '+str(len(inp))+''.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} '+str(len(inp))+''.format(sum(total_fidelity)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    
onebitZ=evol_strategy([0])
onebitO=evol_strategy([1])
onebitsimulation(np.random.randint(0,2,1),onebitZ,onebitO)
onebitZZ=evol_strategy([0,0])
onebitOZ=evol_strategy([0,1])
onebitZO=evol_strategy([1,0])
onebitOO=evol_strategy([1,1])
twobitsimulation(np.random.randint(0,2,2),onebitZZ,onebitZO,onebitOZ,onebitOO)
onebitZZZ=evol_strategy([0,0,0])
onebitZZO=evol_strategy([0,0,1])
onebitZOZ=evol_strategy([0,1,0])
onebitZOO=evol_strategy([0,1,1])
onebitOZZ=evol_strategy([1,0,0])
onebitOZO=evol_strategy([1,0,1])
onebitOOZ=evol_strategy([1,1,0])
onebitOOO=evol_strategy([1,1,1])
threebitsimulation(np.random.randint(0,2,3),onebitZZZ,onebitZZO,onebitZOZ,onebitZOO,onebitOZZ,onebitOZO,onebitOOZ,onebitOOO)
onebitZZZZ=evol_strategy([0,0,0,0])
onebitZZZO=evol_strategy([0,0,0,1])
onebitZZOZ=evol_strategy([0,0,1,0])
onebitZZOO=evol_strategy([0,0,1,1])
onebitZOZZ=evol_strategy([0,1,0,0])
onebitZOZO=evol_strategy([0,1,0,1])
onebitZOOZ=evol_strategy([0,1,1,0])
onebitZOOO=evol_strategy([0,1,1,1])
onebitOZZZ=evol_strategy([1,0,0,0])
onebitOZZO=evol_strategy([1,0,0,1])
onebitOZOZ=evol_strategy([1,0,1,0])
onebitOZOO=evol_strategy([1,0,1,1])
onebitOOZZ=evol_strategy([1,1,0,0])
onebitOOZO=evol_strategy([1,1,0,1])
onebitOOOZ=evol_strategy([1,1,1,0])
onebitOOOO=evol_strategy([1,1,1,1])
fourbitsimulation(np.random.randint(0,2,4),onebitZZZZ,onebitZZZO,onebitZZOZ,onebitZZOO,onebitZOZZ,onebitZOZO,onebitZOOZ,onebitZOOO,onebitOZZZ,onebitOZZO,onebitOZOZ,onebitOZOO,onebitOOZZ,onebitOOZO,onebitOOOZ,onebitOOOO)
onemodE=evol_strategy(np.random.randint(0,2,1))
twomodE=evol_strategy(np.random.randint(0,2,2))
threemodE=evol_strategy(np.random.randint(0,2,3))
fourmodE=evol_strategy(np.random.randint(0,2,4))
simulationev(np.random.randint(0,2,1),onemodE)
simulationev(np.random.randint(0,2,2),twomodE)
simulationev(np.random.randint(0,2,3),threemodE)
simulationev(np.random.randint(0,2,4),fourmodE)
