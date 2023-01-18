#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:46:09 2022
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
         self.data0=np.random.randint(0,2,4)
         self.data1 = np.random.randint(0,2,4)
         self.data2 = np.random.randint(0,2,4)
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
             if( len(self.bob_key) == len(self.data2) ):
             # self.done = True
                 print('This is data1 {} and data2 {} and Bob key {}'.format(self.data1,self.data2,self.bob_key))
                 a=[self.bob_key[i]==self.data2[i] for i in range(len(self.bob_key))]
                 a=np.array(a)
                 if a.all():
                 #if( np.array(self.bob_key) == self.data2 ):
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
             if( len(self.bob_key) == len(self.data2) ):
                 # self.done = True
                 print('This is data1 {} and data2 {} bob key {}'.format(self.data1,self.data2,self.bob_key))
                 a=[self.bob_key[i]==self.data2[i] for i in range(len(self.bob_key))]
                 a=np.array(a)
                 if a.all():
                 #if( np.array(self.bob_key) == self.data2 ):
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
     def reset(self,maxm,inputm,encode=encoded,decode=decoded):
         import numpy as np
         self.max_moves = maxm
         # State for alice
         #self.data0=np.random.randint(0,2,2)
         #self.data1 = np.random.randint(0,2,2)
         self.data1=np.array(inputm)
         print(self.data1)
         #self.data2 = np.random.randint(0,2,2)
         ##self.data0=encode(self.data1,len(self.data1))
         #print(self.data0)
         #self.data2=decode(self.data0,len(self.data0))
         #self.data2=self.data1
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
    env=Qprotocol(4)
    inpu=np.random.randint(0,2,4)
    #models = ANN(D, M, K)
    model.set_params(params)
    #models.set_params(params)
#     # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    counterr=0
    state_n=env.reset(4,inpu)
    obs = state_n[0]#np.concatenate((, state_n[1]), axis=None)#state_n#obs[0]
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
        new_state, reward, done,info=env.step(action)
#         #update total reward
        #done=do
        obs=new_state[0]
        #if do:
        #    counterr+=1
        #if counterr%2==0:
        done=done
# #        print(obs)
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
    model1 = ANN(D,M,K)
    model2 = ANN(D,M,K)
    model3 = ANN(D,M,K)
    model4 = ANN(D,M,K)
    model5 = ANN(D,M,K)
    model6 = ANN(D,M,K)
    model7 = ANN(D,M,K)
    model8 = ANN(D,M,K)
    model9 = ANN(D,M,K)
    model10 = ANN(D,M,K)
    model11 = ANN(D,M,K)
    model12 = ANN(D,M,K)
    model13 = ANN(D,M,K)
    model14 = ANN(D,M,K)
    model15 = ANN(D,M,K)
    #model16 = ANN(D,M,K)
    if len(sys.argv) > 1 and sys.argv[1] =='play':
        #play with a saved model
        j = np.load('es_qkprotocol_resultsQchannel[0, 0, 0, 0].npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        a = np.load('es_qkprotocol_resultsQchannel[0, 0, 0, 1].npz')
        best_paramsa = np.concatenate([a['W1'].flatten(), a['b1'], a['W2'].flatten(), a['b2']])
        s = np.load('es_qkprotocol_resultsQchannel[0, 0, 1, 0].npz')
        best_paramsaa = np.concatenate([s['W1'].flatten(), s['b1'], s['W2'].flatten(), s['b2']])
        e = np.load('es_qkprotocol_resultsQchannel[0, 0, 1, 1].npz')
        best_paramsaaa = np.concatenate([e['W1'].flatten(), e['b1'], e['W2'].flatten(), e['b2']])
        ja = np.load('es_qkprotocol_resultsQchannel[0, 1, 0, 0].npz')
        best_paramse = np.concatenate([j['W1'].flatten(), ja['b1'], ja['W2'].flatten(), ja['b2']])
        aa = np.load('es_qkprotocol_resultsQchannel[0, 1, 0, 1].npz')
        best_paramsee = np.concatenate([a['W1'].flatten(), aa['b1'], aa['W2'].flatten(), aa['b2']])
        sa = np.load('es_qkprotocol_resultsQchannel[0, 1, 1, 0].npz')
        best_paramseee = np.concatenate([s['W1'].flatten(), sa['b1'], sa['W2'].flatten(), sa['b2']])
        ea = np.load('es_qkprotocol_resultsQchannel[0, 1, 1, 1].npz')
        best_paramseeee = np.concatenate([e['W1'].flatten(), ea['b1'], ea['W2'].flatten(), ea['b2']])
        # in case intial shapes are not correct
        jj = np.load('es_qkprotocol_resultsQchannel[1, 1, 1, 1].npz')
        best_paramsq = np.concatenate([j['W1'].flatten(), jj['b1'], jj['W2'].flatten(), jj['b2']])
        aa = np.load('es_qkprotocol_resultsQchannel[1, 0, 0, 0].npz')
        best_paramsaq = np.concatenate([a['W1'].flatten(), aa['b1'], aa['W2'].flatten(), aa['b2']])
        ss = np.load('es_qkprotocol_resultsQchannel[1, 0, 0, 1].npz')
        best_paramsaaq = np.concatenate([s['W1'].flatten(), ss['b1'], ss['W2'].flatten(), ss['b2']])
        ee = np.load('es_qkprotocol_resultsQchannel[1, 0, 1, 0].npz')
        best_paramsaaaq = np.concatenate([e['W1'].flatten(), ee['b1'], ee['W2'].flatten(), ee['b2']])
        jae = np.load('es_qkprotocol_resultsQchannel[1, 0, 1, 1].npz')
        best_paramseq = np.concatenate([j['W1'].flatten(), jae['b1'], jae['W2'].flatten(), jae['b2']])
        aae = np.load('es_qkprotocol_resultsQchannel[1, 1, 0, 0].npz')
        best_paramseeq = np.concatenate([a['W1'].flatten(), aae['b1'], aae['W2'].flatten(), aae['b2']])
        sae = np.load('es_qkprotocol_resultsQchannel[1, 1, 0, 1].npz')
        best_paramseeeq = np.concatenate([s['W1'].flatten(), sae['b1'], sae['W2'].flatten(), sae['b2']])
        saea = np.load('es_qkprotocol_resultsQchannel[1, 1, 1, 0].npz')
        best_paremseeeq = np.concatenate([s['W1'].flatten(), saea['b1'], saea['W2'].flatten(), saea['b2']])

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
        #model1.init()
        #model2.init()
        #model3.init()
        #model4.init()
        #model5.init()
        #model6.init()
        #model7.init()
        #model8.init()
        #model9.init()
        #model10.init()
        #model11.init()
        #model12.init()
        #model13.init()
        #model14.init()
        #model15.init()
        #model16.init()
        params = model.get_params()
        best_params, rewards, actions, learn_r,sigmv,pop_s,parms = evolution_strategy(
            f=reward_function,
            population_size=500,
            sigma=0.002,
            lr=0.006,
            initial_params=params,
            num_iters=400,
        )
            
        model.set_params(best_params)
        model1.set_params(best_paramsa)
        model2.set_params(best_paramsaa)
        model3.set_params(best_paramsaaa)
        model4.set_params(best_paramse)
        model5.set_params(best_paramsee)
        model6.set_params(best_paramseee)
        model7.set_params(best_paramseeee)
        model8.set_params(best_paramsq)
        model9.set_params(best_paramsaq)
        model10.set_params(best_paramsaaq)
        model11.set_params(best_paramsaaaq)
        model12.set_params(best_paramseq)
        model13.set_params(best_paramseeq)
        model14.set_params(best_paramseeeq)
        model15.set_params(best_paremseeeq)
        np.savez('es_qkprotocol_results111100.npz',
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
        total_episodes0=[]
        total_episodes1=[]
        total_episodes2=[]
        total_episodes3=[]
        total_episodes4=[]
        total_episodes5=[]
        total_episodes6=[]
        total_episodes7=[]
        total_episodes8=[]
        total_episodes9=[]
        total_episodes10=[]
        total_episodes11=[]
        total_episodes12=[]
        total_episodes13=[]
        total_episodes14=[]
        total_episodes15=[]
        solved=0
        episodes=100
        Rewa=0
        cum_re=[]
        cum_re1=[]
        total_ep=[]
        steps_ep=[]
        cumre=0
        for _ in range(episodes):
            Rew0, ac0,steps0=reward_function(best_params)
            Rew1, ac1,steps1=reward_function(best_paramsa)
            Rew2, ac2,steps2=reward_function(best_paramsaa)
            Rew3, ac3,steps3=reward_function(best_paramsaaa)
            Rew4, ac4,steps4=reward_function(best_paramsaaa)
            Rew5, ac5,steps5=reward_function(best_paramsaaa)
            Rew6, ac6,steps6=reward_function(best_paramsaaa)
            Rew7, ac7,steps7=reward_function(best_paramsaaa)
            Rew8, ac8,steps8=reward_function(best_paramsq)
            Rew9, ac9,steps9=reward_function(best_paramsaq)
            Rew10, ac10,steps10=reward_function(best_paramsaaq)
            Rew11, ac11,steps11=reward_function(best_paramsaaaq)
            Rew12, ac12,steps12=reward_function(best_paramseeq)
            Rew13, ac13,steps13=reward_function(best_paramseeeq)
            Rew14, ac14,steps14=reward_function(best_paremseeeq)
            Rew15, ac15,steps15=reward_function(best_paramsaaaq)
            #print('This is the reward {}'.format(Rew0,Rew1,Rew2,Rew3))
            #print(steps)
            Rewa+=Rew
            total_episodes0.append(Rew0)
            Rewa += total_episodes0[-1]
            total_episodes1.append(Rew1)
            Rewa += total_episodes1[-1]
            total_episodes2.append(Rew2)
            Rewa += total_episodes2[-1]
            total_episodes3.append(Rew3)
            Rewa += total_episodes3[-1]
            total_episodes4.append(Rew4)
            Rewa += total_episodes4[-1]
            total_episodes5.append(Rew5)
            Rewa += total_episodes5[-1]
            total_episodes6.append(Rew6)
            Rewa += total_episodes6[-1]
            total_episodes7.append(Rew7)
            Rewa += total_episodes7[-1]
            total_episodes8.append(Rew8)
            Rewa += total_episodes8[-1]
            total_episodes9.append(Rew9)
            Rewa += total_episodes9[-1]
            total_episodes10.append(Rew10)
            Rewa += total_episodes10[-1]
            total_episodes11.append(Rew11)
            Rewa += total_episodes11[-1]
            total_episodes12.append(Rew12)
            Rewa += total_episodes12[-1]
            total_episodes13.append(Rew13)
            Rewa += total_episodes13[-1]
            total_episodes14.append(Rew14)
            Rewa += total_episodes14[-1]
            total_episodes15.append(Rew15)
            Rewa += total_episodes15[-1]
            cum_re.append(Rewa)
            #print(ac)
            steps_ep.append(steps0)
            steps_ep.append(steps1)
            steps_ep.append(steps2)
            steps_ep.append(steps3)
            steps_ep.append(steps4)
            steps_ep.append(steps5)
            steps_ep.append(steps6)
            steps_ep.append(steps7)
            steps_ep.append(steps8)
            steps_ep.append(steps9)
            steps_ep.append(steps10)
            steps_ep.append(steps11)
            steps_ep.append(steps12)
            steps_ep.append(steps13)
            steps_ep.append(steps14)
            steps_ep.append(steps15)
            if Rew0>0 or Rew1>0 or Rew2>0 or Rew3>0 or Rew4>0 or Rew5>0 or Rew6>0 or Rew7>0 or Rew8>0 or Rew9>0 or Rew10>0 or Rew11>0 or Rew12>0 or Rew13>0 or Rew15>0:
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re1.append(cumre)
            else:
                total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re1)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(min(steps_ep),len(x))
    #steps=np.arange(0,len(steps_ep),min(steps_ep))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {}'.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    #x=np.arange(0,len(j['learning_rate_v']))
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{}'.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))