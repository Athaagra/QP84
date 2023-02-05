#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:35:33 2022
@author: Optimus
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
#from tensorflow import keras

callbacks=tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True#,
    #start_from_epoch=0
    )

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
                     print("Alice tried to read more bits than available")
                 else:
                     print('This the input message data1 {}'.format(self.data1))
                     self.alice_datalog.append(self.data1[self.alice_data_counter])
                     print('This is alice datalog:{}'.format(self.alice_datalog))
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
     def reset(self,maxm):
         import numpy as np
         self.max_moves = maxm
         # State for alice
         #self.data0=np.random.randint(0,2,2)
         #self.data1 = np.random.randint(0,2,2)
         print('this is the bitstring message {} and the target message {}'.format(self.data1,self.data2))
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
# =============================================================================



EPISODES=0
#random.seed(0)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 1e-3
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.01
        self.learning_rate = 1e-3#20.25
        self.q_value=[]
        self.q_value_pr=[]
        self.model = self._build_model()
        #self.modelT = self._build_model(4)
    
    def _build_model(self):
        model = Sequential()
        #24
        model.add(Dense(self.action_size, input_dim=self.state_size, activation='softmax'))
        #model.add(Dense(24, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.add(Dense(self.action_size, activation=''))
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=Adam(lr=self.learning_rate))
        return model 
    
    def memorize(self,state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        #act_valuesT = self.modelT.predict(state)
        return np.argmax(act_values[0])#,np.argmax(act_valuesT[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        co=0
        ho=[]
        for state, action, reward, next_state, done in minibatch:
            target=reward
            targetT=reward
            if not done:
                target = (reward+self.gamma*np.amax(self.model.predict(next_state)[0]))
                #targetT = (reward+self.gamma*np.amax(self.modelT.predict(next_state)[0]))
                #print(target)
                #artat=np.argmax(target)
                #print(artat)
                self.q_value.append(target)
            target_f=self.model.predict(state)
            co+=1
            #target_fT=self.modelT.predict(state)
            #print(target_f)
            artatt=np.argmax(target_f)
            #print(artatt)
            self.q_value_pr.append(target_f[0][artatt])
            #print('This is the target {}'.format(target))
            #print('This is the target f {} action {}'.format(target_f[0],action))
            target_f[0][action]=target
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            #target_fT[0][action[1]]=target
            history = self.model.fit(state, target_f,epochs=10, batch_size=32, callbacks=[callback],verbose=0)
            #history=self.model.fit(state, target_f, epochs=1,verbose=0,batch_size=batch_size)#,callbacks=[callback])
            #history=self.modelT.fit(state, target_fT, epochs=1,verbose=0,batch_size=batch_size,callbacks=[callback])
            #print(history.history['loss'])
        print(history.history)
        ho.append(history)
        import matplotlib as mpl
        #mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig4=plt.figure(tight_layout=True)
        error = history.history['loss']
        accuracy = history.history['accuracy']
        plt.plot(accuracy)
        plt.plot(error)
        plt.ylabel('loss')
        plt.xlabel('Epochs\nloss train_set={:0.4f}'.format(history.history['loss'][-1], history.history['accuracy'][-1]))
        #plt.legend(['acc_train', 'acc_test'])
        #plt.savefig('accuracy.png')
        plt.show()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history
            
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
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
            resultss.append(['Qlearning p-value We accept the null hypothesis:',pvalue])
        else:
            print("The p-value is less than we reject the null hypothesis")
            resultss.append(['Qlearning p-value the p-value is less than we reject the null hypothesis:',pvalue])
    else:
        print('identical')
        pvalue=0
    plt.figure(figsize=(13, 13))
    plt.bar(1,pvalue)
    plt.xlabel(f'Mannwhitney Test')
    plt.ylabel('Probability')
    plt.title(str(resultss))#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return resultss

def Dqn(inpu,ag,ma,qp):
    state_size=4
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    action_size=len(actions_list)
    batch_size=32
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    acl=[]
    total_episodes=[]
    total_fidelity=[]
    r=0
    EPISODES=100
    cumulative_reward=[]
    for e in range(EPISODES):
        qpO=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state_n,inp=qpO.reset(4)
        state=state_n
        steps_ep=0
        done=False
        reward_episode=[]
        state = np.array(state[0])
        state=np.reshape(state, [1, state_size])
        while done!= True:
            actiona=ag.act(state)
            actiona=np.array(actiona)
            action = actions_list[actiona]
            new_state,reward,done,info,bob_key=qpO.step(action)
            steps_ep+=1
            next_state=np.array(new_state[0])
            next_state= np.reshape(next_state, [1, state_size])
            ag.memorize(state, actiona, reward, next_state, done)
            print("acc-lo {}".format(acl))
            state = next_state
            reward_episode.append(reward)
            if done:
                print('This is the input {} and bob key {}'.format(inp,bob_key))
                bob_key=bob_key[:len(inp)]
                if len(inp)==1 and len(bob_key)==len(inp):
                    tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    steps_epi.append(steps_ep)
                    total_fidelity.append(Fidelity)
                if len(inp)==2 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    steps_epi.append(steps_ep)
                    total_fidelity.append(Fidelity)
                if len(inp)==3 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    steps_epi.append(steps_ep)
                    total_fidelity.append(Fidelity)
                if len(inp)==4 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    steps_epi.append(steps_ep)
                    total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward==1:
                    solved+=1                
                    print("episode : {}/{},reward {}".format(e, EPISODES,reward))#, solved, agent.epsilon,reward))
                    break 
                if len(ag.memory) > batch_size:
                    h= ag.replay(batch_size)
                    acl.append(h)
                    
                    qval.append(ag.q_value)
                    qval_pr.append(ag.q_value_pr)
        ag.save("./QP84DQNd1"+str(inpu)+"CPD.h5")
        r+=reward_episode[-1]
        cumulative_reward.append(r)
        total_episodes.append(reward_episode[-1])
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(inp)+' Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('rewardDQN'+str(inpu)+'.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning cumulative:{}'.format(max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('cmumlaDQN'+str(inpu)+'.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('stepsDQN'+str(inpu)+'.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity per episode :{}'.format(sum(total_fidelity)))
    plt.savefig('fdlyDQN'+str(inpu)+'.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(qval[0])
    plt.plot(qval_pr[0])
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Q value')
    plt.title('The Q value')#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=qpO.error_counter
    results=mannwhitney(total_episodes,error)
    results.append(['Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return ag,results

def Dqnsimulation(inpu,ag,ma,qp):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    cumre=0
    cumulative_reward=[]
    total_fidelity=[]
    reward_episode=[]
    for e in range(EPISODES):
            env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
            state_n,inp=env.reset(4)
            steps_ep=0
            done=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            while done!=True:
                actiona=ag.act(state)
                actiona=np.array(actiona)
                actionA = actions_list[actiona]
                state,reward,done,action_h,bob_key=env.step(actionA)
                steps_ep+=1
                next_state=np.array(state[0])
                next_state= np.reshape(next_state, [1, state_size])
                state = next_state
                if done==True:
                    print('This is bob_key{} inp {}'.format(bob_key,inp))
                    bk=bob_key[:len(inp)]
                    steps_epi.append(steps_ep)
                    if len(inp)==1 and len(inp)==len(bk):
                        tp=LogicalStates[:,inp].T*LogicalStates[bk,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(inp)==len(bk):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bk[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(inp)==len(bk):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bk[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(inp)==len(bk):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bk[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    else:
                        total_fidelity.append(0)
                    if reward==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                        break
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
                        break
            total_episodes.append(reward_episode)
    total_episodes=reward_episode
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('reward.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('cumla.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('fdely.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.savefig('e.png')
    plt.show()
    error=env.error_counter
    results=mannwhitney(total_episodes,error)
    results.append(['Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
def onebitsimulation(ma,ag,ag1,qp):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    cumre=0
    cumulative_reward=[]
    total_fidelity=[]
    for e in range(EPISODES):
            inp=np.random.randint(0,2,1)
            env=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            env1=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            state_n,inp=env.reset(4)
            state_n1,inp=env1.reset(4)
            steps_ep=0
            reward_episode=[]
            done=False
            done1=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            state1 = np.array(state_n1[0])
            state1=np.reshape(state1, [1, state_size])
            while done!=True or done1!=True:
                actiona=ag.act(state)
                actionb=ag1.act(state1)
                actiona=np.array(actiona)
                actionb=np.array(actionb)
                actionA = actions_list[actiona]
                actionB = actions_list[actionb]
                state,reward,done,action_h,bob_key=env.step(actionA)
                state1,reward1,done1,action_h1,bob_key1=env1.step(actionB)
                steps_ep+=1
                print('This is the episode {} done agent 1 {} done agent 2 {}'.format(e,done,done1))
                next_state=np.array(state[0])
                next_state= np.reshape(next_state, [1, state_size])
                next_state1=np.array(state1[0])
                next_state1= np.reshape(next_state1, [1, state_size])
                state = next_state
                state1 = next_state1
                if reward==1:
                    bk=bob_key
                if reward1==1:
                    bk=bob_key1
                if done==True or done1==True:
                    steps_epi.append(steps_ep)
                    print('This is bob key {} and input {}'.format(bk,inp))
                    #if len(bk)<len(inp):
                    tp=LogicalStates[:,inp].T*LogicalStates[bk,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)                        #break
                    #else:
                    #    total_fidelity.append(0)
                    if reward==1 or reward1==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                        break
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
                        break
            total_episodes.append(reward_episode)
    total_episodes=reward_episode
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(cumulative_reward[-1]))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results.append([results1,'Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def twobitsimulation(ma,ag,ag1,ag2,ag3,qp):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    cumre=0
    reward_episode=[]
    cumulative_reward=[]
    total_fidelity=[]
    for e in range(EPISODES):
            inp=np.random.randint(0,2,2)
            env=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            env1=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            env2=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            env3=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
            state_n,inp=env.reset(4)
            state_n1,inp=env1.reset(4)
            state_n2,inp=env2.reset(4)
            state_n3,inp=env3.reset(4)
            steps_ep=0
            done=False
            done1=False
            done2=False
            done3=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            state1 = np.array(state_n1[0])
            state1=np.reshape(state1, [1, state_size])
            state2 = np.array(state_n2[0])
            state2=np.reshape(state2, [1, state_size])
            state3 = np.array(state_n3[0])
            state3=np.reshape(state3, [1, state_size])
            while done!=True or done1!=True or done2!=True or done3!=True:
                actiona=ag.act(state)
                actionb=ag1.act(state1)
                actionc=ag2.act(state2)
                actiond=ag3.act(state3)
                actiona=np.array(actiona)
                actionb=np.array(actionb)
                actionc=np.array(actionc)
                actiond=np.array(actiond)
                actionA = actions_list[actiona]
                actionB = actions_list[actionb]
                actionC = actions_list[actionc]
                actionD = actions_list[actiond]
                state,reward,done,action_h,bob_key=env.step(actionA)
                state1,reward1,done1,action_h1,bob_key1=env1.step(actionB)
                state2,reward2,done2,action_h2,bob_key2=env2.step(actionC)
                state3,reward3,done3,action_h3,bob_key3=env3.step(actionD)
                steps_ep+=1
                next_state=np.array(state[0])
                next_state= np.reshape(next_state, [1, state_size])
                next_state1=np.array(state1[0])
                next_state1= np.reshape(next_state1, [1, state_size])
                next_state2=np.array(state2[0])
                next_state2= np.reshape(next_state2, [1, state_size])
                next_state3=np.array(state3[0])
                next_state3= np.reshape(next_state3, [1, state_size])
                state = next_state
                state1 = next_state1
                state2 = next_state2
                state3 = next_state3
                if done:
                    bk=bob_key
                if done1:
                    bk=bob_key1
                if done2:
                    bk=bob_key2
                if done3:
                    bk=bob_key3
                if done or done1 or done2 or done3:
                    if len(bk)==len(inp):
                        print('This is the input {} and bob key {}'.format(inp,bk))
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bk[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                        steps_epi.append(steps_ep)
                    else:
                        total_fidelity.append(0)
                        steps_epi.append(steps_ep)
                    if reward==1 or reward1==1 or reward2==1 or reward3==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                        break
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
                        break
            total_episodes.append(reward_episode)
    total_episodes=reward_episode
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(cumulative_reward[-1]))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results.append([results1,results2,results3,'Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
def threebitsimulation(ma,ag,ag1,ag2,ag3,ag4,ag5,ag6,ag7,qp):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    cumre=0
    cumulative_reward=[]
    total_fidelity=[]
    reward_episode=[]
    for e in range(EPISODES):
            inp=np.random.randint(0,2,3)
            env=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env1=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env2=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env3=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env4=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env5=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env6=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            env7=Qprotocol(4,inp,Qb=qp,MultiAgent=ma)
            state_n,inp=env.reset(4)
            state_n1,inp=env1.reset(4)
            state_n2,inp=env2.reset(4)
            state_n3,inp=env3.reset(4)
            state_n4,inp=env4.reset(4)
            state_n5,inp=env5.reset(4)
            state_n6,inp=env6.reset(4)
            state_n7,inp=env7.reset(4)
            steps_ep=0
            done=False
            done1=False
            done2=False
            done3=False
            done4=False
            done5=False
            done6=False
            done7=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            state1 = np.array(state_n1[0])
            state1=np.reshape(state1, [1, state_size])
            state2 = np.array(state_n2[0])
            state2=np.reshape(state2, [1, state_size])
            state3 = np.array(state_n3[0])
            state3=np.reshape(state3, [1, state_size])
            state4 = np.array(state_n4[0])
            state4=np.reshape(state4, [1, state_size])
            state5 = np.array(state_n5[0])
            state5=np.reshape(state5, [1, state_size])
            state6 = np.array(state_n6[0])
            state6=np.reshape(state6, [1, state_size])
            state7 = np.array(state_n7[0])
            state7=np.reshape(state7, [1, state_size])
            while done!=True or done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=0:
                actiona=ag.act(state)
                actionb=ag1.act(state1)
                actionc=ag2.act(state2)
                actiond=ag3.act(state3)
                actione=ag4.act(state4)
                actionf=ag5.act(state5)
                actiong=ag6.act(state6)
                actionh=ag7.act(state7)
                actiona=np.array(actiona)
                actionb=np.array(actionb)
                actionc=np.array(actionc)
                actiond=np.array(actiond)
                actione=np.array(actione)
                actionf=np.array(actionf)
                actiong=np.array(actiong)
                actionh=np.array(actionh)
                actionA = actions_list[actiona]
                actionB = actions_list[actionb]
                actionC = actions_list[actionc]
                actionD = actions_list[actiond]
                actionE = actions_list[actione]
                actionF = actions_list[actionf]
                actionG = actions_list[actiong]
                actionH = actions_list[actionh]
                state,reward,done,action_h,bob_key=env.step(actionA)
                state1,reward1,done1,action_h1,bob_key1=env1.step(actionB)
                state2,reward2,done2,action_h2,bob_key2=env2.step(actionC)
                state3,reward3,done3,action_h3,bob_key3=env3.step(actionD)
                state4,reward4,done4,action_h4,bob_key4=env4.step(actionE)
                state5,reward5,done5,action_h5,bob_key5=env5.step(actionF)
                state6,reward6,done6,action_h6,bob_key6=env6.step(actionG)
                state7,reward7,done7,action_h7,bob_key7=env7.step(actionH)
                steps_ep+=1
                next_state=np.array(state[0])
                next_state= np.reshape(next_state, [1, state_size])
                next_state1=np.array(state1[0])
                next_state1= np.reshape(next_state1, [1, state_size])
                next_state2=np.array(state2[0])
                next_state2= np.reshape(next_state2, [1, state_size])
                next_state3=np.array(state3[0])
                next_state3= np.reshape(next_state3, [1, state_size])
                next_state4=np.array(state4[0])
                next_state4= np.reshape(next_state4, [1, state_size])
                next_state5=np.array(state5[0])
                next_state5= np.reshape(next_state5, [1, state_size])
                next_state6=np.array(state6[0])
                next_state6= np.reshape(next_state6, [1, state_size])
                next_state7=np.array(state7[0])
                next_state7= np.reshape(next_state7, [1, state_size])
                state = next_state
                state1 = next_state1
                state2 = next_state2
                state3 = next_state3
                state4 = next_state4
                state5 = next_state5
                state6 = next_state6
                state7 = next_state7
                print(reward,reward1,reward2,reward3,reward4,reward5,reward6,reward7)
                if done or done1 or done2 or done3 or done4 or done5 or done6 or done7:
                    if done:
                        bk=bob_key
                    if done1:
                        bk=bob_key1
                    if done2:
                        bk=bob_key2
                    if done3:
                        bk=bob_key3
                    if done4:
                        bk=bob_key4
                    if done5:
                        bk=bob_key5
                    if done6:
                        bk=bob_key6
                    if done7:
                        bk=bob_key7
                    print('This is the input {} and bob key {}'.format(inp,bk))
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bk[:len(inp)])
                    if len(bob_keys)!=len(inp):
                        total_fidelity.append(0)
                    else:
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    steps_epi.append(steps_ep)
                    if reward==1 or reward1==1 or reward2==1 or reward3==1 or reward4==1 or reward5==1 or reward6==1 or reward7==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                        break
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
                        break
            total_episodes.append(reward_episode)
    total_episodes=reward_episode
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(cumulative_reward[-1]))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
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
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results4=mannwhitney(total_episodes,error4)
    results5=mannwhitney(total_episodes,error5)
    results6=mannwhitney(total_episodes,error6)
    results7=mannwhitney(total_episodes,error7)
    results.append([results1,results2,results3,results4,results5,results6,results7,'Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def fourbitsimulation(ma,ag,ag1,ag2,ag3,ag4,ag5,ag6,ag7,ag8,ag9,ag10,ag11,ag12,ag13,ag14,ag15,qp):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    r=0
    cumulative_reward=[]
    total_fidelity=[]
    reward_episode=[]
    for e in range(EPISODES):
            inpu=np.random.randint(0,2,4)
            env=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env1=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env2=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env3=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env4=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env5=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env6=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env7=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env8=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env9=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env10=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env11=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env12=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env13=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env14=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            env15=Qprotocol(4,inpu,Qb=qp,MultiAgent=ma)
            state1,inpu=env.reset(4)
            state2,inpu=env1.reset(4)
            state3,inpu=env2.reset(4)
            state4,inpu=env3.reset(4)
            state5,inpu=env4.reset(4)
            state6,inpu=env5.reset(4)
            state7,inpu=env6.reset(4)
            state8,inpu=env7.reset(4)
            state9,inpu=env8.reset(4)
            state10,inpu=env9.reset(4)
            state11,inpu=env10.reset(4)
            state12,inpu=env11.reset(4)
            state13,inpu=env12.reset(4)
            state14,inpu=env13.reset(4)
            state15,inpu=env14.reset(4)
            state16,inpu=env15.reset(4)
            steps_ep=0
            done1=False
            done2=False
            done3=False
            done4=False
            done5=False
            done6=False
            done7=False
            done8=False
            done9=False
            done10=False
            done11=False
            done12=False
            done13=False
            done14=False
            done15=False
            done16=False
            state1 = np.array(state1[0])
            state1=np.reshape(state1, [1, state_size])
            state2 = np.array(state2[0])
            state2=np.reshape(state2, [1, state_size])
            state3 = np.array(state3[0])
            state3=np.reshape(state3, [1, state_size])
            state4 = np.array(state4[0])
            state4=np.reshape(state4, [1, state_size])
            state5 = np.array(state5[0])
            state5=np.reshape(state5, [1, state_size])
            state6 = np.array(state6[0])
            state6=np.reshape(state6, [1, state_size])
            state7 = np.array(state7[0])
            state7=np.reshape(state7, [1, state_size])
            state8 = np.array(state8[0])
            state8=np.reshape(state8, [1, state_size])
            state9 = np.array(state9[0])
            state9=np.reshape(state9, [1, state_size])
            state10 = np.array(state10[0])
            state10=np.reshape(state10, [1, state_size])
            state11 = np.array(state11[0])
            state11=np.reshape(state11, [1, state_size])
            state12 = np.array(state12[0])
            state12=np.reshape(state12, [1, state_size])
            state13 = np.array(state13[0])
            state13=np.reshape(state13, [1, state_size])
            state14 = np.array(state14[0])
            state14=np.reshape(state14, [1, state_size])
            state15 = np.array(state15[0])
            state15=np.reshape(state15, [1, state_size])
            state16 = np.array(state16[0])
            state16=np.reshape(state16, [1, state_size]) 
            while done1!=True or done2!=True or done3!=True or done8!=True or done5!=True or done6!=True or done7!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
                actiona=ag.act(state1)
                actionb=ag1.act(state2)
                actionc=ag2.act(state3)
                actiond=ag3.act(state4)            
                actione=ag4.act(state5)
                actionf=ag5.act(state6)
                actiong=ag6.act(state7)
                actionh=ag7.act(state8)
                actioni=ag8.act(state9)
                actionk=ag9.act(state10)
                actionl=ag10.act(state11)
                actionm=ag11.act(state12)            
                actionn=ag12.act(state13)
                actionp=ag13.act(state14)
                actionq=ag14.act(state15)
                actionr=ag15.act(state16)
                actiona=np.array(actiona)
                actionb=np.array(actionb)
                actionc=np.array(actionc)
                actiond=np.array(actiond)
                actione=np.array(actione)
                actionf=np.array(actionf)
                actiong=np.array(actiong)
                actionh=np.array(actionh)
                actioni=np.array(actioni)
                actionk=np.array(actionk)
                actionl=np.array(actionl)
                actionm=np.array(actionm)
                actionn=np.array(actionn)
                actionp=np.array(actionp)
                actionq=np.array(actionq)
                actionr=np.array(actionr)
                actionA = actions_list[actiona]
                actionB = actions_list[actionb]
                actionC = actions_list[actionc]
                actionD = actions_list[actiond]
                actionE = actions_list[actione]
                actionF = actions_list[actionf]
                actionG = actions_list[actiong]
                actionH = actions_list[actionh]
                actionI = actions_list[actioni]
                actionK = actions_list[actionk]
                actionM = actions_list[actionl]
                actionN = actions_list[actionm]
                actionO = actions_list[actionn]
                actionP = actions_list[actionp]
                actionQ = actions_list[actionq]
                actionR = actions_list[actionr]
                state1,reward1,done1,action_h1,bob_key1=env.step(actionA)
                state2,reward2,done2,action_h2,bob_key2=env1.step(actionB)
                state3,reward3,done3,action_h3,bob_key3=env2.step(actionC)
                state4,reward4,done4,action_h4,bob_key4=env3.step(actionD)
                state5,reward5,done5,action_h5,bob_key5=env4.step(actionE)
                state6,reward6,done6,action_h6,bob_key6=env5.step(actionF)
                state7,reward7,done7,action_h7,bob_key7=env6.step(actionG)
                state8,reward8,done8,action_h8,bob_key8=env7.step(actionH)
                state9,reward9,done9,action_h9,bob_key9=env8.step(actionI)
                state10,reward10,done10,action_h10,bob_key10=env9.step(actionK)
                state11,reward11,done11,action_h11,bob_key11=env10.step(actionM)
                state12,reward12,done12,action_h12,bob_key12=env11.step(actionN)
                state13,reward13,done13,action_h13,bob_key13=env12.step(actionO)
                state14,reward14,done14,action_h14,bob_key14=env13.step(actionP)
                state15,reward15,done15,action_h15,bob_key15=env14.step(actionQ)
                state16,reward16,done16,action_h16,bob_key16=env15.step(actionR)
                if done1==True:
                    bk=bob_key1
                if done2==True:
                    bk=bob_key2
                if done3==True:
                    bk=bob_key3
                if done4==True:
                    bk=bob_key4
                if done5==True:
                    bk=bob_key5
                if done6==True:
                    bk=bob_key6
                if done7==True:
                    bk=bob_key7
                if done8==True:
                    bk=bob_key8
                if done9==True:
                    bk=bob_key9
                if done10==True:
                    bk=bob_key10
                if done11==True:
                    bk=bob_key11
                if done12==True:
                    bk=bob_key12
                if done13==True:
                    bk=bob_key13
                if done14==True:
                    bk=bob_key14
                if done15==True:
                    bk=bob_key15
                if done16==True:
                    bk=bob_key16
                steps_ep+=1
                next_state1=np.array(state1[0])
                next_state1= np.reshape(next_state1, [1, state_size])
                next_state2=np.array(state2[0])
                next_state2= np.reshape(next_state2, [1, state_size])
                next_state3=np.array(state3[0])
                next_state3= np.reshape(next_state3, [1, state_size])
                next_state4=np.array(state4[0])
                next_state4= np.reshape(next_state4, [1, state_size])
                next_state5=np.array(state5[0])
                next_state5= np.reshape(next_state5, [1, state_size])
                next_state6=np.array(state6[0])
                next_state6= np.reshape(next_state6, [1, state_size])
                next_state7=np.array(state7[0])
                next_state7= np.reshape(next_state7, [1, state_size])
                next_state8=np.array(state8[0])
                next_state8= np.reshape(next_state8, [1, state_size])
                next_state9=np.array(state9[0])
                next_state9= np.reshape(next_state9, [1, state_size])
                next_state10=np.array(state10[0])
                next_state10= np.reshape(next_state10, [1, state_size])
                next_state11=np.array(state11[0])
                next_state11= np.reshape(next_state11, [1, state_size])
                next_state12=np.array(state12[0])
                next_state12= np.reshape(next_state12, [1, state_size])
                next_state13=np.array(state13[0])
                next_state13= np.reshape(next_state13, [1, state_size])
                next_state14=np.array(state14[0])
                next_state14= np.reshape(next_state14, [1, state_size])
                next_state15=np.array(state15[0])
                next_state15= np.reshape(next_state15, [1, state_size])
                next_state16=np.array(state16[0])
                next_state16= np.reshape(next_state16, [1, state_size])
                state1 = next_state1
                state2 = next_state2
                state3 = next_state3
                state4 = next_state4
                state5 = next_state5
                state6 = next_state6
                state7 = next_state7
                state8 = next_state8
                state9 = next_state9
                state10 = next_state10
                state11 = next_state11
                state12 = next_state12
                state13 = next_state13
                state14 = next_state14
                state15 = next_state15
                state16 = next_state16
                print(reward1,reward2,reward3,reward4,reward5,reward6,reward7,reward8,reward9,reward10,reward11,reward12,reward13,reward14,reward15,reward16)
                if done1 or done2 or done3 or done4 or done5 or done6 or done7 or done8 or done9 or done10 or done11 or done12 or done13 or done14 or done15 or done16:
                    inpus=''.join(str(x) for x in inpu)
                    bob_keys=''.join(str(x) for x in bk[:len(inpu)])
                    if len(bk)==len(inpu):
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    else:
                        total_fidelity.append(0)
                    steps_epi.append(steps_ep)
                    if reward1==1 or reward2==1 or reward3==1 or reward4==1 or reward5==1 or reward6==1 or reward7==1 or reward8==1 or reward9==1 or reward10==1 or reward11==1 or reward12==4 or reward13==1 or reward14==1 or reward15==1 or reward16==1:
                        solved+=1
                        print('The reward is {}'.format(solved))
                        r+=1
                        reward_episode.append(1)
                        cumulative_reward.append(r)
                        break
                    else:
                        r-=1
                        reward_episode.append(0)
                        cumulative_reward.append(r)
                        break 
            total_episodes.append(reward_episode)
    total_episodes=reward_episode
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(cumulative_reward[-1]))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
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
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results4=mannwhitney(total_episodes,error4)
    results5=mannwhitney(total_episodes,error5)
    results6=mannwhitney(total_episodes,error6)
    results7=mannwhitney(total_episodes,error7)
    results8=mannwhitney(total_episodes,error8)
    results9=mannwhitney(total_episodes,error9)
    results10=mannwhitney(total_episodes,error10)
    results11=mannwhitney(total_episodes,error11)
    results12=mannwhitney(total_episodes,error12)
    results13=mannwhitney(total_episodes,error13)
    results14=mannwhitney(total_episodes,error14)
    results15=mannwhitney(total_episodes,error15)
    results.append([results1,results2,results3,results4,results5,results6,results7,results8,results9,results10,results11,results12,results13,results14,results15,'Reward:'+str(solved/EPISODES),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
state_size=4
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
action_size=len(actions_list)#env.action_space.n

agent = DQNAgent(state_size, action_size)
agent,r=Dqn(1,agent,False,False)
print(r,file=open('randomOneBitDqnTraining.txt','w'))
r=Dqnsimulation(1,agent,False,False)
print(r,file=open('randomOneBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(2,agent,False,False)
print(r,file=open('randomTwoBitDqnTraining.txt','w'))
Dqnsimulation(2,agent,False,False)
print(r,file=open('randomTwoBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(3,agent,False,False)
print(r,file=open('randomThreeBitDqnTraining.txt','w'))
r=Dqnsimulation(3,agent,False,False)
print(r,file=open('randomThreeBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(4,agent,False,False)
print(r,file=open('randomFourBitDqnTraining.txt','w'))
r=Dqnsimulation(4,agent,False,False)
print(r,file=open('randomFourBitDqnTesting.txt','w'))


agent = DQNAgent(state_size, action_size)
agent,r=Dqn(1,agent,False,True)
print(r,file=open('randomOneQBitDqnTraining.txt','w'))
r=Dqnsimulation(1,agent,False,True)
print(r,file=open('randomOneQBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(2,agent,False,True)
print(r,file=open('randomTwoQBitDqnTraining.txt','w'))
Dqnsimulation(2,agent,False,True)
print(r,file=open('randomTwoQBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(3,agent,False,True)
print(r,file=open('randomThreeQBitDqnTraining.txt','w'))
r=Dqnsimulation(3,agent,False,True)
print(r,file=open('randomThreeQBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(4,agent,False,True)
print(r,file=open('randomFourQBitDqnTraining8.txt','w'))
r=Dqnsimulation(4,agent,False,True)
print(r,file=open('randomFourQBitDqnTesting.txt','w'))



##Training
agent = DQNAgent(state_size, action_size)
agent,r=Dqn([0],agent,True,False)
print(r,file=open('randomOne[0]BitDqnTraining.txt','w'))
agentO = DQNAgent(state_size, action_size)
agentO,r=Dqn([1],agentO,True,False)
print(r,file=open('randomOne[1]BitDqnTraining.txt','w'))
r=onebitsimulation(True,agent,agentO,False)
print(r,file=open('randomOneBitMULTIDqnTesting.txt','w'))


agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQNd1[0]CPD.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNd1[1]CPD.h5")
r=onebitsimulation(1,False,agent,agent1,False)
print(r,file='randomOneBitMULTIDqnTesting.txt')


agent = DQNAgent(state_size, action_size)
agent,r=Dqn([0,0],agent,True,False)
print(r,file=open('randomTwo[0,0]BitDqnTraining.txt','w'))
agentO = DQNAgent(state_size, action_size)
agentO,r=Dqn([0,1],agentO,True,False)
print(r,file=open('randomTwo[0,1]BitDqnTraining.txt','w'))
agentT = DQNAgent(state_size, action_size)
agentT,r=Dqn([1,0],agentT,True,False)
print(r,file=open('randomTwo[1,0]BitDqnTraining.txt','w'))
agentTh = DQNAgent(state_size, action_size)
agentTh,r=Dqn([1,1],agentTh,True,False)
print(r,file=open('randomTwo[1,1]BitDqnTraining.txt','w'))
r=twobitsimulation(True,agent,agentO,agentT,agentTh,False)
print(r,file=open('randomTwoMULTIBitDqnTraining.txt','w'))

agent3 = DQNAgent(state_size, action_size)
agent3.load("./QP84DQNd1[0,0]CPD.h5")
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQNd1[0,1]CPD.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNd1[1,0]CPD.h5")
agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQNd1[1,1]CPD.h5")
r=twobitsimulation(agent,agent1,agent2,agent3)
print(r,file='randomTwoBitMULTIDqnTesting.txt')


agent=DQNAgent(state_size, action_size)
agent,r=Dqn([0,0,0],agent,True,False)
print(r,file=open('randomThree[0,0,0]BitDqnTraining.txt','w'))
agentO=DQNAgent(state_size, action_size)
agentO,r=Dqn([0,0,1],agentO,True,False)
print(r,file=open('randomThree[0,0,1]BitDqnTraining.txt','w'))
agentT=DQNAgent(state_size, action_size)
agentT,r=Dqn([0,1,0],agentT,True,False)
print(r,file=open('randomThree[0,1,0]BitDqnTraining.txt','w'))
agentTh=DQNAgent(state_size, action_size)
agentTh,r=Dqn([0,1,1],agentTh,True,False)
print(r,file=open('randomThree[0,1,1]BitDqnTraining.txt','w'))
agentFo=DQNAgent(state_size, action_size)
agentFo,r=Dqn([1,0,0],agentFo,True,False)
print(r,file=open('randomThree[1,0,0]BitDqnTraining.txt','w'))
agentFi=DQNAgent(state_size, action_size)
agentFi,r=Dqn([1,0,1],agentFi,True,False)
print(r,file=open('randomThree[1,0,1]BitDqnTraining.txt','w'))
agentSi=DQNAgent(state_size, action_size)
agentSi,r=Dqn([1,1,0],agentSi,True,False)
print(r,file=open('randomThree[1,1,0]BitDqnTraining.txt','w'))
agentSe=DQNAgent(state_size, action_size)
agentSe,r=Dqn([1,1,1],agentSe,True,False)
print(r,file=open('randomThree[1,1,1]BitDqnTraining.txt','w'))
r=threebitsimulation(True,agent,agentO,agentT,agentTh,agentFo,agentFi,agentSi,agentSe,False)
print(r,file=open('randomThreeMULTIBitDqnTesting.txt','w'))

agent7 = DQNAgent(state_size, action_size)
agent7.load("./QP84DQNd1[1, 1, 1]CPD.h5")
agent6 = DQNAgent(state_size, action_size)
agent6.load("./QP84DQNd1[1, 1, 0]CPD.h5")
agent5 = DQNAgent(state_size, action_size)
agent5.load("./QP84DQNd1[1, 0, 1]CPD.h5")
agent4 = DQNAgent(state_size, action_size)
agent4.load("./QP84DQNd1[1, 0, 0]CPD.h5")
agent3 = DQNAgent(state_size, action_size)
agent3.load("./QP84DQNd1[0, 1, 1]CPD.h5")
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQNd1[0, 1, 0]CPD.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNd1[0, 0, 1]CPD.h5")
agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQNd1[0, 0, 0]CPD.h5")
r=r=threebitsimulation(True,agent,agent1,agent2,agent3,agent4,agent5,agent6,agent7,False)
print(r,file=open('randomThreeMULTIBitDqnTesting.txt','w'))

agent=DQNAgent(state_size, action_size)
agent,r=Dqn([0,0,0,0],agent,True,False)
print(r,file=open('randomFour[0,0,0,0]BitDqnTraining.txt','w'))
agentO=DQNAgent(state_size, action_size)
agentO,r=Dqn([0,0,0,1],agentO,True,False)
print(r,file=open('randomFour[0,0,0,1]BitDqnTraining.txt','w'))
agentT=DQNAgent(state_size, action_size)
agentT,r=Dqn([0,0,1,0],agentT,True,False)
print(r,file=open('randomFour[0,0,1,0]BitDqnTraining.txt','w'))
agentTh=DQNAgent(state_size, action_size)
agentTh,r=Dqn([0,0,1,1],agentTh,True,False)
print(r,file=open('randomFour[0,0,1,1]BitDqnTraining.txt','w'))
agentFo=DQNAgent(state_size, action_size)
agentFo,r=Dqn([0,1,0,0],agentFo,True,False)
print(r,file=open('randomFour[0,1,0,0]BitDqnTraining.txt','w'))
agentFi=DQNAgent(state_size, action_size)
agentFi,r=Dqn([0,1,0,1],agentFi,True,False)
print(r,file=open('randomFour[0,1,0,1]BitDqnTraining.txt','w'))
agentSi=DQNAgent(state_size, action_size)
agentSi,r=Dqn([0,1,1,0],agentSi,True,False)
print(r,file=open('randomFour[0,1,1,0]BitDqnTraining.txt','w'))
agentSe=DQNAgent(state_size, action_size)
agentSe,r=Dqn([0,1,1,1],agentSe,True,False)
print(r,file=open('randomFour[0,1,1,1]BitDqnTraining.txt','w'))
agentEi=DQNAgent(state_size, action_size)
agentEi,r=Dqn([1,0,0,0],agentEi,True,False)
print(r,file=open('randomFour[1,0,0,0]BitDqnTraining.txt','w'))
agentN=DQNAgent(state_size, action_size)
agentN,r=Dqn([1,0,0,1],agentN,True,False)
print(r,file=open('randomFour[1,0,0,1]BitDqnTraining.txt','w'))
agentTe=DQNAgent(state_size, action_size)
agentTe,r=Dqn([1,0,1,0],agentTe,True,False)
print(r,file=open('randomFour[1,0,1,0]BitDqnTraining.txt','w'))
agentEl=DQNAgent(state_size, action_size)
agentEl,r=Dqn([1,0,1,1],agentEl,True,False)
print(r,file=open('randomFour[1,0,1,1]BitDqnTraining.txt','w'))
agentTw=DQNAgent(state_size, action_size)
agentTw,r=Dqn([1,1,0,0],agentTw,True,False)
print(r,file=open('randomFour[1,1,0,0]BitDqnTraining.txt','w'))
agentThr=DQNAgent(state_size, action_size)
agentThr,r=Dqn([1,1,0,1],agentThr,True,False)
print(r,file=open('randomFour[1,1,0,1]BitDqnTraining.txt','w'))
agentFor=DQNAgent(state_size, action_size)
agentFor,r=Dqn([1,1,1,0],agentFor,True,False)
print(r,file=open('randomFour[1,1,1,0]BitDqnTraining.txt','w'))
agentFif=DQNAgent(state_size, action_size)
agentFif,r=Dqn([1,1,1,1],agentFif,True,False)
print(r,file=open('randomFour[1,1,1,1]BitDqnTraining.txt','w'))
r=fourbitsimulation(True,agent,agentO,agentT,agentTh,agentFo,agentFi,agentSi,agentSe,agentEi,agentN,agentTe,agentEl,agentTw,agentThr,agentFor,agentFif,False)
print(r,file=open('randomFourMULTIBitDqnTesting.txt','w'))

agent15 = DQNAgent(state_size, action_size)
agent15.load("./QP84DQN0000.h5")
agent14 = DQNAgent(state_size, action_size)
agent14.load("./QP84DQN0001.h5")
agent13 = DQNAgent(state_size, action_size)
agent13.load("./QP84DQN0010.h5")
agent12 = DQNAgent(state_size, action_size)
agent12.load("./QP84DQN0011.h5")
agent11 = DQNAgent(state_size, action_size)
agent11.load("./QP84DQN0100.h5")
agent10 = DQNAgent(state_size, action_size)
agent10.load("./QP84DQN0101.h5")
agent9 = DQNAgent(state_size, action_size)
agent9.load("./QP84DQN0110.h5")
agent8 = DQNAgent(state_size, action_size)
agent8.load("./QP84DQN0111.h5")
agent7 = DQNAgent(state_size, action_size)
agent7.load("./QP84DQN1000.h5")
agent6 = DQNAgent(state_size, action_size)
agent6.load("./QP84DQN1001.h5")
agent5 = DQNAgent(state_size, action_size)
agent5.load("./QP84DQN1010.h5")
agent4 = DQNAgent(state_size, action_size)
agent4.load("./QP84DQN1011.h5")
agent3 = DQNAgent(state_size, action_size)
agent3.load("./QP84DQN1100.h5")
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQN1101.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQN1110.h5")
agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQN1.h5")
r=fourbitsimulation(agent,agent1,agent2,agent3,agent4,agent5,agent6,agent7,agent8,agent9,agent10,agent11,agent12,agent13,agent14,agent15)
print(r,file=open('randomFourMULTIQBitDqnTraining.txt','w'))

agent = DQNAgent(state_size, action_size)
agent,r=Dqn([0],agent,True,True)
print(r,file=open('randomOne[0]QBitDqnTraining.txt','w'))
agentO = DQNAgent(state_size, action_size)
agentO,r=Dqn([1],agentO,True,True)
print(r,file=open('randomOne[1]QBitDqnTraining.txt','w'))
r=onebitsimulation(True,agent,agentO,True)
print(r,file=open('randomOneQBitMULTIQDqnTesting.txt','w'))

agent = DQNAgent(state_size, action_size)
agent,r=Dqn([0,0],agent,True,True)
print(r,file=open('randomTwo[0,0]QBitDqnTraining.txt','w'))
agentO = DQNAgent(state_size, action_size)
agentO,r=Dqn([0,1],agentO,True,True)
print(r,file=open('randomTwo[0,1]QBitDqnTraining.txt','w'))
agentT = DQNAgent(state_size, action_size)
agentT,r=Dqn([1,0],agentT,True,True)
print(r,file=open('randomTwo[1,0]QBitDqnTraining.txt','w'))
agentTh = DQNAgent(state_size, action_size)
agentTh,r=Dqn([1,1],agentTh,True,True)
print(r,file=open('randomTwo[1,1]QBitDqnTraining.txt','w'))
r=twobitsimulation(True,agent,agentO,agentT,agentTh,True)
print(r,file=open('randomTwoMULTIQBitDqnTraining.txt','w'))

agent=DQNAgent(state_size, action_size)
agent,r=Dqn([0,0,0],agent,True,True)
print(r,file=open('randomThree[0,0,0]QBitDqnTraining.txt','w'))
agentO=DQNAgent(state_size, action_size)
agentO,r=Dqn([0,0,1],agentO,True,True)
print(r,file=open('randomThree[0,0,1]QBitDqnTraining.txt','w'))
agentT=DQNAgent(state_size, action_size)
agentT,r=Dqn([0,1,0],agentT,True,True)
print(r,file=open('randomThree[0,1,0]QBitDqnTraining.txt','w'))
agentTh=DQNAgent(state_size, action_size)
agentTh,r=Dqn([0,1,1],agentTh,True,True)
print(r,file=open('randomThree[0,1,1]QBitDqnTraining.txt','w'))
agentFo=DQNAgent(state_size, action_size)
agentFo,r=Dqn([1,0,0],agentFo,True,True)
print(r,file=open('randomThree[1,0,0]QBitDqnTraining.txt','w'))
agentFi=DQNAgent(state_size, action_size)
agentFi,r=Dqn([1,0,1],agentFi,True,True)
print(r,file=open('randomThree[1,0,1]QBitDqnTraining.txt','w'))
agentSi=DQNAgent(state_size, action_size)
agentSi,r=Dqn([1,1,0],agentSi,True,True)
print(r,file=open('randomThree[1,1,0]QBitDqnTraining.txt','w'))
agentSe=DQNAgent(state_size, action_size)
agentSe,r=Dqn([1,1,1],agentSe,True,True)
print(r,file=open('randomThree[1,1,1]QBitDqnTraining.txt','w'))
r=threebitsimulation(True,agent,agentO,agentT,agentTh,agentFo,agentFi,agentSi,agentSe,True)
print(r,file=open('randomThreeMULTIQBitDqnTesting.txt','w'))

agent=DQNAgent(state_size, action_size)
agent,r=Dqn([0,0,0,0],agent,True,True)
print(r,file=open('randomFour[0,0,0,0]QBitDqnTraining.txt','w'))
agentO=DQNAgent(state_size, action_size)
agentO,r=Dqn([0,0,0,1],agentO,True,True)
print(r,file=open('randomFour[0,0,0,1]QBitDqnTraining.txt','w'))
agentT=DQNAgent(state_size, action_size)
agentT,r=Dqn([0,0,1,0],agentT,True,True)
print(r,file=open('randomFour[0,0,1,0]QBitDqnTraining.txt','w'))
agentTh=DQNAgent(state_size, action_size)
agentTh,r=Dqn([0,0,1,1],agentTh,True,True)
print(r,file=open('randomFour[0,0,1,1]QBitDqnTraining.txt','w'))
agentFo=DQNAgent(state_size, action_size)
agentFo,r=Dqn([0,1,0,0],agentFo,True,True)
print(r,file=open('randomFour[0,1,0,0]QBitDqnTraining.txt','w'))
agentFi=DQNAgent(state_size, action_size)
agentFi,r=Dqn([0,1,0,1],agentFi,True,True)
print(r,file=open('randomFour[0,1,0,1]QBitDqnTraining.txt','w'))
agentSi=DQNAgent(state_size, action_size)
agentSi,r=Dqn([0,1,1,0],agentSi,True,True)
print(r,file=open('randomFour[0,1,1,0]QBitDqnTraining.txt','w'))
agentSe=DQNAgent(state_size, action_size)
agentSe,r=Dqn([0,1,1,1],agentSe,True,True)
print(r,file=open('randomFour[0,1,1,1]QBitDqnTraining.txt','w'))
agentEi=DQNAgent(state_size, action_size)
agentEi,r=Dqn([1,0,0,0],agentEi,True,True)
print(r,file=open('randomFour[1,0,0,0]QBitDqnQTraining.txt','w'))
agentN=DQNAgent(state_size, action_size)
agentN,r=Dqn([1,0,0,1],agentN,True,True)
print(r,file=open('randomFour[1,0,0,1]QBitDqnQTraining.txt','w'))
agentTe=DQNAgent(state_size, action_size)
agentTe,r=Dqn([1,0,1,0],agentTe,True,True)
print(r,file=open('randomFour[1,0,1,0]QBitDqnQTraining.txt','w'))
agentEl=DQNAgent(state_size, action_size)
agentEl,r=Dqn([1,0,1,1],agentEl,True,True)
print(r,file=open('randomFour[1,0,1,1]QBitDqnQTraining.txt','w'))
agentTw=DQNAgent(state_size, action_size)
agentTw,r=Dqn([1,1,0,0],agentTw,True,True)
print(r,file=open('randomFour[1,1,0,0]QBitDqnQTraining.txt','w'))
agentThr=DQNAgent(state_size, action_size)
agentThr,r=Dqn([1,1,0,1],agentThr,True,True)
print(r,file=open('randomFour[1,1,0,1]QBitDqnQTraining.txt','w'))
agentFor=DQNAgent(state_size, action_size)
agentFor,r=Dqn([1,1,1,0],agentFor,True,True)
print(r,file=open('randomFour[1,1,1,0]QBitDqnQTraining.txt','w'))
agentFif=DQNAgent(state_size, action_size)
agentFif,r=Dqn([1,1,1,1],agentFif,True,True)
print(r,file=open('randomFour[1,1,1,1]QBitDqnQTraining.txt','w'))
r=fourbitsimulation(True,agent,agentO,agentT,agentTh,agentFo,agentFi,agentSi,agentSe,agentEi,agentN,agentTe,agentEl,agentTw,agentThr,agentFor,agentFif,True)
print(r,file=open('randomFourMULTIQBitDqnTesting.txt','w'))
