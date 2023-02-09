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
import sys

#temp = sys.stdout                 # store original stdout object for later
#sys.stdout = open('log.txt', 'w')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:46:53 2023

@author: Optimus
"""
"""
The environment for Level1
# Actions for Alice:
# 0 - Idle
# 1 - Read next bit from data1, store in datalog
# 2 - Place datalog in Bob's mailbox
# Actions for Bob:
# 0 - Idle
# 1 - Read next bit from mailbox
# 2 - Write 0 to key
# 3 - Write 1 to key
# Actions are input to the environment as tuples
# e.g. (1,0) means Alice takes action 1 and Bob takes action 0
# Rewards accumulate: negative points for wrong guess, positive points for correct guess
# Game terminates with correct key or N moves
# """
import numpy as np
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
     def __init__(self,maxm,inp,encode=encoded,decode=decoded,Qb=False,MultiAgent=False):
         self.max_moves = maxm
         if MultiAgent==True:
             self.data1=inp
             self.data0=np.random.randint(0,2,len(inp))
             self.data2 = np.random.randint(0,2,len(inp))
         else:
             self.data0=np.random.randint(0,2,inp)
             self.data1 = np.random.randint(0,2,inp)
             self.data2 = np.random.randint(0,2,inp)
         if Qb==True:
             self.data0=encode(self.data1,len(self.data1))
         #print(self.data0)
             self.data2=decode(self.data0,len(self.data0))
         ####Classical Channel
         else:
             self.data2=self.data1
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
         self.reset(maxm)
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
                     print('')
                     #print("Alice tried to read more bits than available")
                 else:
                     print('This the input message data1 {}'.format(self.data1))
                     self.alice_datalog.append(self.data1[self.alice_data_counter])
                     print('This is alice datalog:{}'.format(self.alice_datalog))
                     self.alice_data_counter += 1
                 if verbose:
                     print('')
                     #print("Alice added data1 to the datalog ", self.alice_datalog)
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
                         print('')
                         #print("Bob tried to read more bits than available")
                     else:
                         self.bob_datalog[self.bob_data_counter % len(self.bob_datalog)] = self.bob_mailbox[self.bob_data_counter]
                         self.bob_data_counter += 1
         if verbose:
             print('')
             #print("Bob added to his datalog ", self.bob_datalog)
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
     def reset(self,maxm):
         import numpy as np
         self.max_moves = maxm
         # State for alice
         #self.data0=np.random.randint(0,2,2)
         #self.data1 = np.random.randint(0,2,2)
         #print('this is the bitstring message {} and the target message {}'.format(self.data1,self.data2))
         #self.data1=self.data1#np.array(np.random.randint(0,2,inputm))
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
         return state,self.data1
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

def mannwhitney(total_episodes,error):
    from scipy.stats import mannwhitneyu
    # seed the random number generator
    resultss=[]
    if sum(total_episodes)!=sum(error):
        stat, pvalue = mannwhitneyu(total_episodes, error)
        print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
        # interpret
        if pvalue > 0.05:
            print('We accept the null hypothesis')
            resultss.append(['evolutionary strategy p-value We accept the null hypothesis:',pvalue])
        else:
            print("The p-value is less than we reject the null hypothesis")
            resultss.append(['evolutionary strategy p-value the p-value is less than we reject the null hypothesis:',pvalue])
    else:
        print('identical')
        pvalue=0
    #x=1
    #plt.figure(figsize=(13, 13))
    #print('This is pvalue {}'.format(pvalue))
    #plt.bar(x,pvalue)
    #plt.xlabel(f'Mannwhitney Test')
    #plt.ylabel('Probability')
    #plt.title(str(resultss))#.format(solved/EPISODES))
    #plt.grid(True,which="both",ls="--",c='gray')
    #plt.show()
    return resultss
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
        
def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters,inputme,qp,ma):
    #assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    learning_rate = np.zeros(num_iters)
    sigma_v = np.zeros(num_iters)
    parms = np.zeros(num_iters)
    params = initial_params
    envprotocol=Qprotocol(4,inputme,Qb=qp,MultiAgent=ma)
    total_fidelity=[]
    cum_re=[]
    cumre=0
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        ### slow way
        R = np.zeros(population_size)
        acts=np.zeros(population_size)
        total_fidelity=[]
        steps_epi=[]
        total_ep=[]
        solved=0
        #print('This is the number of acts {}'.format(len(acts)))
        #loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j],acts[j],_,bk,d,inputme = f(params_try,envprotocol)
            #print('This is action {} and Reward {}'.format(acts[j],R[j]))
            if d==True:
                if len(bk)==len(inputme) and len(inputme)==1:
                    tp=LogicalStates[:,inputme].T*LogicalStates[bk,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    if R[j]==1: 
                        solved+=1
                        cumre+=1
                        total_ep.append(1)
                        #cum_re.append(cumre)
                    else:
                        cumre+=0
                        solved+=0
                        total_ep.append(0)
                        #cum_re.append(cumre)
                if len(bk)==len(inputme) and len(inputme)==2:
                    inpus=''.join(str(x) for x in inputme)
                    bob_keys=''.join(str(x) for x in bk[:len(inputme)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    if R[j]==1: 
                        solved+=1
                        cumre+=1
                        total_ep.append(1)
                        #cum_re.append(cumre)
                    else:
                        cumre+=0
                        solved+=0
                        total_ep.append(0)
                        #cum_re.append(cumre)
                if len(bk)==len(inputme) and len(inputme)==3:
                    inpus=''.join(str(x) for x in inputme)
                    bob_keys=''.join(str(x) for x in bk[:len(inputme)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    if R[j]==1: 
                        solved+=1
                        cumre+=1
                        total_ep.append(1)
                        #cum_re.append(cumre)
                    else:
                        cumre+=0
                        solved+=0
                        total_ep.append(0)
                        #cum_re.append(cumre)
                if len(bk)==len(inputme) and len(inputme)==4:
                    inpus=''.join(str(x) for x in inputme)
                    bob_keys=''.join(str(x) for x in bk[:len(inputme)])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    if R[j]==1: 
                        solved+=1
                        cumre+=1
                        total_ep.append(1)
                        #cum_re.append(cumre)
                    else:
                        cumre+=0
                        solved+=0
                        total_ep.append(0)
                        #cum_re.append(cumre)
        cum_re.append(cumre)
        error=envprotocol.error_counter
        resu=mannwhitney(R,error)
        #print(resu)
        #print('This is solved {}'.format(solved))
        #print('This is population size {}'.format(population_size))
        print('This is the cumulative reward {}'.format(cumre))
        #print('this is the number of steps {}'.format(steps_epi))
        #print('this is the number of steps {}'.format(total_fidelity))
        #resu.append(['Reward:'+str(solved/population_size),'Cumulative:'+str(cumre),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
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
        #print("Iter:",t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:",(datetime.now()-t0))
        if m > 0.01:# or R.max() >= 1:#m > R.max()/1.5 or R.max() >= 1:
            actis = acts
            print('True')
            #break
        else:
            actis=np.zeros(population_size)
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel('Number of episode {}'.format(max(cum_re)))
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return params, reward_per_iteration,actis,learning_rate,sigma_v,population_size,parms,resu
    
def reward_function(params,envprotocol):
    model = ANN(D, M, K)
    model.set_params(params)
    episode_reward = 0
    episode_length = 0
    done = False
    counterr=0
    state_n,inpu=envprotocol.reset(4)
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
        new_state, reward, done,info,bob_key=envprotocol.step(action)
        obs=new_state[0]
        episode_reward += reward
        episode_length +=1
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona,episode_length,bob_key,done,inpu
#
def evol_strategy(mes,qp,ma,sigma_parameter=0.002,population_size_parameter=5000,learning_rate=0.006,number_of_iterations=500):     
    model = ANN(D,M,K)
    model.init()
    params = model.get_params()
    message=mes
    best_params, rewards, actions, learn_r,sigmv,pop_s,parms,results = evolution_strategy(
                f=reward_function,
                population_size=population_size_parameter,
                sigma=sigma_parameter,
                lr=learning_rate,
                initial_params=params,
                num_iters=number_of_iterations,
                inputme=message,
                qp=qp,
                ma=ma,)
    np.savez('es_qkprotocol_results'+str(message)+'.npz',
             learning_rate_v=learn_r,
             sigmav=sigmv,
             populat_s=pop_s,
             pmeters=parms,
             actions_e=actions,
             train=rewards,
             **model.get_params_dict(),)
    return best_params,results

def simulationev(inpa,bp,qp,ma):
    total_fidelity=[]
    solved=0
    episodes=100
    cum_re=[]
    total_ep=[]
    steps_ep=[]
    steps_epi=[]
    cumre=0
    for _ in range(episodes):
        env=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        Rew,ac,steps,bobk,d,inp=reward_function(bp,env)
        bk=bobk
        inputm=inp
        print(inp,bobk,bk,steps,Rew)
        steps_ep.append(steps)
        if len(inp)==1 and len(bk)==len(inp):
            tp=LogicalStates[:,inputm].T*LogicalStates[bk,:]
            tp=tp[0]
            Fidelity=abs(sum(tp))**2
            total_fidelity.append(Fidelity)
#            if Rew==1: 
#                solved+=1
#                cumre+=1
#                total_ep.append(1)
#                cum_re.append(cumre)
#            else:
#                cumre+=0
#                solved+=0
#                total_ep.append(0)
#                cum_re.append(cumre)
        if len(inp)==2 and len(bk)==len(inp):
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            total_fidelity.append(Fidelity)
#            if Rew==1: 
#                solved+=1
#                cumre+=1
#                total_ep.append(1)
#                cum_re.append(cumre)
#            else:
#                cumre+=0
#                solved+=0
#                total_ep.append(0)
#                cum_re.append(cumre)
        if len(inp)==3 and len(bk)==len(inp):
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            total_fidelity.append(Fidelity)
#        if Rew==1: 
 #               solved+=1
 #               cumre+=1
 #               total_ep.append(1)
 #               cum_re.append(cumre)
 #           else:
 #               cumre+=0
 #               solved+=0
 #               total_ep.append(0)
 #               cum_re.append(cumre)
        if len(inp)==4 and len(bk)==len(inp):
            inpus=''.join(str(x) for x in inp)
            bob_keys=''.join(str(x) for x in bk[:len(inp)])
            tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
            Fidelity=abs(sum(tp))**2
            total_fidelity.append(Fidelity)
        else:
            total_fidelity.append(0)
        if Rew==1: 
            solved+=1
            cumre+=1
            total_ep.append(1)
            cum_re.append(cumre)
        else:
            cumre+=0
            solved+=0
            total_ep.append(0)
            cum_re.append(cumre)
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    print(cum_re)
    plt.title('Cumulative Reward {}'.format(max(cum_re)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    plt.plot(x,steps_ep)
    plt.title('Number of steps per episode {} {}'.format(np.mean(steps_ep),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} {}'.format(solved/episodes,len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} {}'.format(sum(total_fidelity),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    print('This is total episode {},{} error {},{}'.format(total_ep,len(total_ep),error,len(error)))
    results=mannwhitney(total_ep,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(cum_re[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results


def onebitsimulation(inpn,bp,bp1,qp,ma):
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
        inpa=np.random.randint(0,2,inpn)
        env=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env1=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        Rew,ac,steps,bobk,d,inp=reward_function(bp,env)
        Rew1,ac1,steps1,bobk1,d1,inp=reward_function(bp1,env1)
        if Rew==1:
            bk=bobk
            inputm=inp
            steps_ep.append(steps)
        if Rew1==1:
            bk=bobk1
            inputm=inp
            steps_ep.append(steps1)
        if len(inp)==len(bobk) and len(inp)==1 or len(inp)==len(bobk1) and len(inp)==1 and len(bk)!=0:
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
            solved+=0
            cumre+=0
            cum_re.append(cumre)
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
    #steps=np.repeat(steps_ep,len(x))
    plt.plot(x,steps_ep)
    plt.title('Number of steps per episode {} {}'.format(np.mean(steps_ep),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} {}'.format(solved/episodes,str(len(inp))))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} {}'.format(sum(total_fidelity),str(len(inp))))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    results=mannwhitney(total_ep,error)
    results1=mannwhitney(total_ep,error1)
    results.append([results1,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum_re[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def twobitsimulation(inpa,bp,bp1,bp2,bp3,qp,ma):
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
        inpa=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env1=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env2=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env3=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        Rew, ac,steps,bobk,d,inp=reward_function(bp,env)
        Rew1, ac1,steps1,bobk1,d1,inp=reward_function(bp1,env1)
        Rew2, ac2,steps2,bobk2,d2,inp=reward_function(bp2,env2)
        Rew3, ac3,steps3,bobk3,d3,inp=reward_function(bp3,env3)
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
            solved+=0
            cumre+=0
            cum_re.append(cumre)
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_epi))
    #steps=np.repeat(steps_ep,len(x))
    plt.plot(x,steps_epi)
    plt.title('Number of steps per episode {} {}'.format(np.mean(steps),str(len(inp))))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} {}'.format(solved/episodes,len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} {}'.format(sum(total_fidelity),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    print(total_ep,error)
    results=mannwhitney(total_ep,error)
    print(total_ep,error1)
    results1=mannwhitney(total_ep,error1)
    print(total_ep,error2)
    results2=mannwhitney(total_ep,error2)
    print(total_ep,error3)
    results3=mannwhitney(total_ep,error3)
    results.append([results1,results2,results3,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum_re[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
def threebitsimulation(inpa,bp,bp1,bp2,bp3,bp4,bp5,bp6,bp7,ma,qp):
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
        inpa=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env1=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env2=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env3=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env4=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env5=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env6=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        env7=Qprotocol(4,inpa,Qb=qp,Multiagent=ma)
        Rew, ac,steps,bobk,d,inp=reward_function(bp,env)
        Rew1, ac1,steps1,bobk1,d1,inp=reward_function(bp1,env1)
        Rew2, ac2,steps2,bobk2,d2,inp=reward_function(bp2,env2)
        Rew3, ac3,steps3,bobk3,d3,inp=reward_function(bp3,env3)
        Rew4, ac4,steps4,bobk4,d4,inp=reward_function(bp4,env4)
        Rew5, ac5,steps5,bobk5,d5,inp=reward_function(bp5,env5)
        Rew6, ac6,steps6,bobk6,d6,inp=reward_function(bp6,env6)
        Rew7, ac7,steps7,bobk7,d7,inp=reward_function(bp7,env7)
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
            solved+=0
            cumre+=0
            cum_re.append(cumre)
            total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_epi))
    #steps=np.repeat(steps_epi,len(x))
    plt.plot(x,steps_epi)
    plt.title('Number of steps per episode {} {}'.format(np.mean(steps),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} {}'.format(solved/episodes,len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} {}'.format(sum(total_fidelity),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    error4=env4.error_counter
    error5=env5.error_counter
    error6=env6.error_counter
    error7=env7.error_counter
    results=mannwhitney(total_ep,error)
    results1=mannwhitney(total_ep,error1)
    results2=mannwhitney(total_ep,error2)
    results3=mannwhitney(total_ep,error3)
    results4=mannwhitney(total_ep,error4)
    results5=mannwhitney(total_ep,error5)
    results6=mannwhitney(total_ep,error6)
    results7=mannwhitney(total_ep,error7)
    results.append([results1,results2,results3,results4,results5,results6,results7,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum_re[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
def fourbitsimulation(inpa,bp,bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8,bp9,bp10,bp11,bp12,bp13,bp14,bp15,qp,ma):
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
        inpa=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env1=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env2=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env3=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env4=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env5=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env6=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env7=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env8=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env9=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env10=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env11=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env12=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env13=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env14=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        env15=Qprotocol(4,inpa,Qb=qp,MultiAgent=ma)
        Rew, ac,steps,bobk,d,inp=reward_function(bp,env)
        Rew1, ac1,steps1,bobk1,d1,inp=reward_function(bp1,env1)
        Rew2, ac2,steps2,bobk2,d2,inp=reward_function(bp2,env2)
        Rew3, ac3,steps3,bobk3,d3,inp=reward_function(bp3,env3)
        Rew4, ac4,steps4,bobk4,d4,inp=reward_function(bp4,env4)
        Rew5, ac5,steps5,bobk5,d5,inp=reward_function(bp5,env5)
        Rew6, ac6,steps6,bobk6,d6,inp=reward_function(bp6,env6)
        Rew7, ac7,steps7,bobk7,d7,inp=reward_function(bp7,env7)
        Rew8, ac8,steps8,bobk8,d8,inp=reward_function(bp8,env8)
        Rew9, ac9,steps9,bobk9,d9,inp=reward_function(bp9,env9)
        Rew10,ac10,steps10,bobk10,d10,inp=reward_function(bp10,env10)
        Rew11, ac11,steps11,bobk11,d11,inp=reward_function(bp11,env11)
        Rew12, ac12,steps12,bobk12,d12,inp=reward_function(bp12,env12)
        Rew13, ac13,steps13,bobk13,d13,inp=reward_function(bp13,env13)
        Rew14, ac14,steps14,bobk14,d14,inp=reward_function(bp14,env14)
        Rew15, ac15,steps15,bobk15,d15,inp=reward_function(bp15,env15)
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
            solved+=0
            cumre+=0
            total_ep.append(0)
            cum_re.append(cumre)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re)
    plt.xlabel(f'Number of Steps of episode '+str(len(inp))+'')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_epi))
    #steps=np.repeat(steps_epi,len(x))
    plt.plot(x,steps_epi)
    plt.title('Number of steps per episode {} {}'.format(np.mean(steps),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{} {}'.format(solved/episodes,len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.title('The simulation has been solved the environment Evolutionary Strategy fidelity score:{} {}'.format(sum(total_fidelity),len(inp)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    error4=env4.error_counter
    error5=env5.error_counter
    error6=env6.error_counter
    error7=env7.error_counter
    error8=env8.error_counter
    error9=env9.error_counter
    error10=env10.error_counter
    error11=env11.error_counter
    error12=env12.error_counter
    error13=env13.error_counter
    error14=env14.error_counter
    error15=env15.error_counter
    results=mannwhitney(total_ep,error)
    results1=mannwhitney(total_ep,error1)
    results2=mannwhitney(total_ep,error2)
    results3=mannwhitney(total_ep,error3)
    results4=mannwhitney(total_ep,error4)
    results5=mannwhitney(total_ep,error5)
    results6=mannwhitney(total_ep,error6)
    results7=mannwhitney(total_ep,error7)
    results8=mannwhitney(total_ep,error8)
    results9=mannwhitney(total_ep,error9)
    results10=mannwhitney(total_ep,error10)
    results11=mannwhitney(total_ep,error11)
    results12=mannwhitney(total_ep,error12)
    results13=mannwhitney(total_ep,error13)
    results14=mannwhitney(total_ep,error14)
    results15=mannwhitney(total_ep,error15)
    results.append([results1,results2,results3,results4,results5,results6,results7,results8,results9,results10,results11,results12,results13,results14,results15,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum_re[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
onebitZ,r=evol_strategy([0],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomOneBit[0]EsTraining.txt','w'))
onebitO,r=evol_strategy([1],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomOneBit[1]EsTraining.txt','w'))
r=onebitsimulation(1,onebitZ,onebitO,True,True)
print(r,file=open('randomtOneBitMLTIEsTesting.txt','w'))

onebitZ,r=evol_strategy([0],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomOneQBit[0]EsTraining.txt','w'))
onebitO,r=evol_strategy([1],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomOneQBit[1]EsTraining.txt','w'))
r=onebitsimulation(1,onebitZ,onebitO,True,True)
print(r,file=open('randomtOneQBitMLTIEsTesting.txt','w'))






onebitZZ,r=evol_strategy([0,0],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoBit[0,0]EsTraining.txt','w'))
onebitOZ,r=evol_strategy([0,1],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoBit[0,1]EsTraining.txt','w'))
onebitZO,r=evol_strategy([1,0],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoBit[1,0]EsTraining.txt','w'))
onebitOO,r=evol_strategy([1,1],False,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoBit[1,1]EsTraining.txt','w'))
r=twobitsimulation(2,onebitZZ,onebitZO,onebitOZ,onebitOO,True,True)
print(r,file=open('randomTwoBitMULTIEsTesting.txt','w'))

onebitZZ,r=evol_strategy([0,0],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoQBit[0,0]EsTraining.txt','w'))
onebitOZ,r=evol_strategy([0,1],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoQBit[0,1]EsTraining.txt','w'))
onebitZO,r=evol_strategy([1,0],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoQBit[1,0]EsTraining.txt','w'))
onebitOO,r=evol_strategy([1,1],True,True,sigma_parameter=0.99,learning_rate=1,number_of_iterations=100)
print(r,file=open('randomTwoQBit[1,1]EsTraining.txt','w'))
r=twobitsimulation(2,onebitZZ,onebitZO,onebitOZ,onebitOO,True,True)
print(r,file=open('randomTwoQBitMULTIEsTesting.txt','w'))





onebitZZZ,r=evol_strategy([0,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[0,0,0]EsTraining.txt','w'))
onebitZZO,r=evol_strategy([0,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[0,0,1]EsTraining.txt','w'))
onebitZOZ,r=evol_strategy([0,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[0,1,0]EsTraining.txt','w'))
onebitZOO,r=evol_strategy([0,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[0,1,1]EsTraining.txt','w'))
onebitOZZ,r=evol_strategy([1,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[1,0,0]EsTraining.txt','w'))
onebitOZO,r=evol_strategy([1,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[1,0,1]EsTraining.txt','w'))
onebitOOZ,r=evol_strategy([1,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[1,1,0]EsTraining.txt','w'))
onebitOOO,r=evol_strategy([1,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBit[1,1,1]EsTraining.txt','w'))
r=threebitsimulation(3,onebitZZZ,onebitZZO,onebitZOZ,onebitZOO,onebitOZZ,onebitOZO,onebitOOZ,onebitOOO,False,True)
print(r,file=open('randomThreeBitMULTIEsTesting.txt','w'))

onebitZZZ,r=evol_strategy([0,0,0],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[0,0,0]EsTraining.txt','w'))
onebitZZO,r=evol_strategy([0,0,1],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[0,0,1]EsTraining.txt','w'))
onebitZOZ,r=evol_strategy([0,1,0],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[0,1,0]EsTraining.txt','w'))
onebitZOO,r=evol_strategy([0,1,1],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[0,1,1]EsTraining.txt','w'))
onebitOZZ,r=evol_strategy([1,0,0],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[1,0,0]EsTraining.txt','w'))
onebitOZO,r=evol_strategy([1,0,1],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[1,0,1]EsTraining.txt','w'))
onebitOOZ,r=evol_strategy([1,1,0],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[1,1,0]EsTraining.txt','w'))
onebitOOO,r=evol_strategy([1,1,1],True,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeQBit[1,1,1]EsTraining.txt','w'))
r=threebitsimulation(3,onebitZZZ,onebitZZO,onebitZOZ,onebitZOO,onebitOZZ,onebitOZO,onebitOOZ,onebitOOO,False,True)
print(r,file=open('randomThreeQBitMULTIEsTesting.txt','w'))

      
onebitZZZZ,r=evol_strategy([0,0,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,0,0,0]EsTraining.txt','w'))
onebitZZZO,r=evol_strategy([0,0,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,0,0,1]EsTraining.txt','w'))
onebitZZOZ,r=evol_strategy([0,0,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,0,1,0]EsTraining.txt','w'))
onebitZZOO,r=evol_strategy([0,0,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,0,1,1]EsTraining.txt','w'))
onebitZOZZ,r=evol_strategy([0,1,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,1,0,0]EsTraining.txt','w'))
onebitZOZO,r=evol_strategy([0,1,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,1,0,1]EsTraining.txt','w'))
onebitZOOZ,r=evol_strategy([0,1,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,1,1,0]EsTraining.txt','w'))
onebitZOOO,r=evol_strategy([0,1,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[0,1,1,1]EsTraining.txt','w'))
onebitOZZZ,r=evol_strategy([1,0,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,0,0,0]EsTraining.txt','w'))
onebitOZZO,r=evol_strategy([1,0,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,0,0,1]EsTraining.txt','w'))
onebitOZOZ,r=evol_strategy([1,0,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,0,1,0]EsTraining.txt','w'))
onebitOZOO,r=evol_strategy([1,0,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,0,1,1]EsTraining.txt','w'))
onebitOOZZ,r=evol_strategy([1,1,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,0,0]EsTraining.txt','w'))
onebitOOZO,r=evol_strategy([1,1,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,0,1]EsTraining.txt','w'))
onebitOOOZ,r=evol_strategy([1,1,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,1,0]EsTraining.txt','w'))
onebitOOOO,r=evol_strategy([1,1,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,1,1]EsTraining.txt','w'))
r=fourbitsimulation(4,onebitZZZZ,onebitZZZO,onebitZZOZ,onebitZZOO,onebitZOZZ,onebitZOZO,onebitZOOZ,onebitZOOO,onebitOZZZ,onebitOZZO,onebitOZOZ,onebitOZOO,onebitOOZZ,onebitOOZO,onebitOOOZ,onebitOOOO,False,True)


onebitZZZZ,r=evol_strategy([0,0,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,0,0,0]EsTraining.txt','w'))
onebitZZZO,r=evol_strategy([0,0,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,0,0,1]EsTraining.txt','w'))
onebitZZOZ,r=evol_strategy([0,0,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,0,1,0]EsTraining.txt','w'))
onebitZZOO,r=evol_strategy([0,0,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,0,1,1]EsTraining.txt','w'))
onebitZOZZ,r=evol_strategy([0,1,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,1,0,0]EsTraining.txt','w'))
onebitZOZO,r=evol_strategy([0,1,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,1,0,1]EsTraining.txt','w'))
onebitZOOZ,r=evol_strategy([0,1,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,1,1,0]EsTraining.txt','w'))
onebitZOOO,r=evol_strategy([0,1,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[0,1,1,1]EsTraining.txt','w'))
onebitOZZZ,r=evol_strategy([1,0,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[1,0,0,0]EsTraining.txt','w'))
onebitOZZO,r=evol_strategy([1,0,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[1,0,0,1]EsTraining.txt','w'))
onebitOZOZ,r=evol_strategy([1,0,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[1,0,1,0]EsTraining.txt','w'))
onebitOZOO,r=evol_strategy([1,0,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBit[1,0,1,1]EsTraining.txt','w'))
onebitOOZZ,r=evol_strategy([1,1,0,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,0,0]EsTraining.txt','w'))
onebitOOZO,r=evol_strategy([1,1,0,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,0,1]EsTraining.txt','w'))
onebitOOOZ,r=evol_strategy([1,1,1,0],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,1,0]EsTraining.txt','w'))
onebitOOOO,r=evol_strategy([1,1,1,1],False,True,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBit[1,1,1,1]EsTraining.txt','w'))
r=fourbitsimulation(4,onebitZZZZ,onebitZZZO,onebitZZOZ,onebitZZOO,onebitZOZZ,onebitZOZO,onebitZOOZ,onebitZOOO,onebitOZZZ,onebitOZZO,onebitOZOZ,onebitOZOO,onebitOOZZ,onebitOOZO,onebitOOOZ,onebitOOOO,False,True)






print(r,file=open('randomFourBitMULTIEsTesting.txt','w'))
onemodE,r=evol_strategy(1,False,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomOneBitsEsTraining.txt','w'))
twomodE,r=evol_strategy(2,False,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomTwoBitsEsTraining.txt','w'))
threemodE,r=evol_strategy(3,False,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomThreeBitsEsTraining.txt','w'))
fourmodE,r=evol_strategy(4,False,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourBitsEsTraining.txt','w'))

r=simulationev(1,onemodE,False,False)
print(r,file=open('randomOneBitsEsTesting.txt','w'))
r=simulationev(2,twomodE,False,False)
print(r,file=open('randomTwoBitsEsTesting.txt','w'))
r=simulationev(3,threemodE,False,False)
print(r,file=open('randomThreeBitsEsTesting.txt','w'))
r=simulationev(4,fourmodE,False,False)
print(r,file=open('randomFourBitsEsTesting.txt','w'))


onemodEQ,r=evol_strategy(1,True,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomOneQBitsEsTraining.txt','w'))
twomodEQ,r=evol_strategy(2,True,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomTwoQBitsEsTraining.txt','w'))
threemodEQ,r=evol_strategy(3,True,False,sigma_parameter=0.99,learning_rate=1e-1,number_of_iterations=500)
print(r,file=open('randomThreeQBitsEsTraining.txt','w'))
fourmodEQ,r=evol_strategy(4,True,False,sigma_parameter=0.99,learning_rate=1e-5,number_of_iterations=100)
print(r,file=open('randomFourQBitsEsTraining.txt','w'))


r=simulationev(1,onemodEQ,True,False)
print(r,file=open('randomOneQBitsEsTesting.txt','w'))
r=simulationev(2,twomodEQ,True,False)
print(r,file=open('randomTwoQBitsEsTesting.txt','w'))
r=simulationev(3,threemodEQ,True,False)
print(r,file=open('randomThreeQBitsEsTesting.txt','w'))
r=simulationev(4,fourmodEQ,True,False)
print(r,file=open('randomFourQBitsEsTesting.txt','w'))

