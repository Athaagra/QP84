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
#import environment.ClassicalCommunicationChNumberOfActions as Cch
import environment.QprotocolEncodindDecodingNumberOfActions as Qch

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
#D = 4#len(env.reset())*HISTORY_LENGTH
D=8
M = 32
#K = 12
K=30
#actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]

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
        
def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters,inputml,qp,ma):
    #assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    learning_rate = np.zeros(num_iters)
    sigma_v = np.zeros(num_iters)
    parms = np.zeros(num_iters)
    params = initial_params
    #envprotocol=Qprotocol(4,inputme,Qb=qp,MultiAgent=ma)
    #envprotocol=Qch.Qprotocol(8,inputme,MultiAgent=ma)
    total_fidelity=[]
    cum_re=[]
    cumre=0
    for t in range(num_iters):
        #t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        print('This is inputme {}'.format(inputml))
        envprotocol=Qch.Qprotocol(8,inputml,MultiAgent=ma)
        print('This is the number of iteration {}'.format(t))
        ### slow way
        R = np.zeros(population_size)
        acts=np.zeros(population_size)
        #total_fidelity=[]
        #steps_epi=[]
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
        #print('This is the cumulative reward {}'.format(cumre))
        #print('this is the number of steps {}'.format(steps_epi))
        #print('this is the number of steps {}'.format(total_fidelity))
        #resu.append(['Reward:'+str(solved/population_size),'Cumulative:'+str(cumre),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
        m = R.mean()
        #resu=0
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
    #state_n,inpu=envprotocol.reset(4)
    #envprotocol=Qch.Qprotocol(8,inptme,MultiAgent=False)
    state_n,inpu=envprotocol.reset(8)
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
                inputml=message,
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
    actions_made_list=[]
    #env=Qch.Qprotocol(8,inpa,MultiAgent=ma)
    cumre=0
    for _ in range(episodes):
        env=Qch.Qprotocol(8,inpa,MultiAgent=ma)
        Rew,ac,steps,bobk,d,inp=reward_function(bp,env)
        actions_made_list.append(ac)
        bk=bobk
        inputm=inp
        print(inp,bobk,bk,steps,Rew)
        steps_ep.append(steps)
        if len(inp)==1 and len(bk)==len(inp):
            tp=LogicalStates[:,inputm].T*LogicalStates[bk,:]
            tp=tp[0][0]
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
        elif len(inp)==2 and len(bk)==len(inp):
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
        elif len(inp)==3 and len(bk)==len(inp):
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
        elif len(inp)==4 and len(bk)==len(inp):
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
    print('This is total fidelity {}'.format(total_fidelity))
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
    print('Those are the actions made {}'.format(actions_made_list))
    return results,actions_made_list
