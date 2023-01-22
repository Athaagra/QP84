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
# =============================================================================



EPISODES=50
#random.seed(0)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.01
        self.learning_rate = 1e-4#20.25
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
            history = self.model.fit(state, target_f,epochs=2, batch_size=32, callbacks=[callback],verbose=0)
            #history=self.model.fit(state, target_f, epochs=1,verbose=0,batch_size=batch_size)#,callbacks=[callback])
            #history=self.modelT.fit(state, target_fT, epochs=1,verbose=0,batch_size=batch_size,callbacks=[callback])
            #print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
LogicalStates=np.array([[1,0],[0,1]])
LogicalStates2bit=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
LogicalStates3bit=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
import pandas as pd
columns2bit=['00','01','10','11']
columns3bit=['000','001','010','011','100','101','110','111']
LogicalStates2bit=pd.DataFrame(LogicalStates2bit, columns=columns2bit)
LogicalStates3bit=pd.DataFrame(LogicalStates3bit, columns=columns3bit)
LogicalStates2bit=LogicalStates2bit.rename(index={0:'00',1:'01',2:'10',3:'11'})
LogicalStates3bit=LogicalStates3bit.rename(index={0:'000',1:'001',2:'010',3:'011',4:'100',5:'101',6:'110',7:'111'})
#if __name__=="__main__":
def Dqn(inp,ag):
    inp=[1]
    state_size=4
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    action_size=len(actions_list)
    batch_size=32
    solved=0
    qpO=Qprotocol(4)
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    total_fidelity=[]
    r=0
    cumulative_reward=[]
    for e in range(EPISODES):
        state_n=qpO.reset(4,inp)
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
            state = next_state
            reward_episode.append(reward)
            if done:
                tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                tp=tp[0]
                #tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                #tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                Fidelity=abs(sum(tp))**2
                steps_epi.append(steps_ep)
                total_fidelity.append(Fidelity)
                if reward==1:
                    solved+=1                
                    print("episode : {}/{},reward {}".format(e, EPISODES,reward))#, solved, agent.epsilon,reward))
                    break 
                if len(ag.memory) > batch_size:
                    ag.replay(batch_size)
                    qval.append(ag.q_value)
                    qval_pr.append(ag.q_value_pr)
                ag.save("./QP84DQNd1"+str(inp)+"CPD.h5")
        r+=reward_episode[-1]
        cumulative_reward.append(r)
        total_episodes.append(reward_episode[-1])
    
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
    plt.title('The simulation has been solved the environment Deep Q learning cumulative:{}'.format(cumulative_reward[-1]))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity per episode :{}'.format(sum(total_fidelity)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(qval[0])
    plt.plot(qval_pr[0])
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Q value')
    plt.title('The Q value')#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return ag

def onebitsimulation(ag,ag1):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    inp=np.random.randint(0,2,1)
    env=Qprotocol(4)
    env1=Qprotocol(4)
    cumre=0
    cumulative_reward=[]
    total_fidelity=[]
    for e in range(EPISODES):
            state_n=env.reset(4,inp)
            state_n1=env1.reset(4,inp)
            steps_ep=0
            reward_episode=[]
            done=False
            done1=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            state1 = np.array(state_n1[0])
            state1=np.reshape(state1, [1, state_size])
            while done!=True or done1!=True:
                actiona=agent.act(state)
                actionb=agent1.act(state1)
                actiona=np.array(actiona)
                actionb=np.array(actionb)
                actionA = actions_list[actiona]
                actionB = actions_list[actionb]
                state,reward,done,action_h,bob_key=env.step(actionA)
                state1,reward1,done1,action_h1,bob_key1=env1.step(actionB)
                steps_ep+=1
                next_state=np.array(state[0])
                next_state= np.reshape(next_state, [1, state_size])
                next_state1=np.array(state1[0])
                next_state1= np.reshape(next_state1, [1, state_size])
                state = next_state
                state1 = next_state1
                if done:
                    bk=bob_key
                if done1:
                    bk=bob_key1
                if done or done1:
                    tp=LogicalStates[:,inp].T*LogicalStates[bk,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    steps_epi.append(steps_ep)
                    if reward==1 or reward1==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
            total_episodes.append(reward_episode)
    
    plt.figure(figsize=(13, 13))
    print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
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
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(total_fidelity))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()


def twobitsimulation(ag,ag1,ag2,ag3):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    inp=np.random.randint(0,2,1)
    env=Qprotocol(4)
    env1=Qprotocol(4)
    env2=Qprotocol(4)
    env3=Qprotocol(4)
    cumre=0
    cumulative_reward=[]
    total_fidelity=[]
    for e in range(EPISODES):
            state_n=env.reset(4,inp)
            state_n1=env1.reset(4,inp)
            state_n2=env2.reset(4,inp)
            state_n3=env3.reset(4,inp)
            steps_ep=0
            reward_episode=[]
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
                actiona=agent.act(state)
                actionb=agent1.act(state1)
                actionc=agent2.act(state2)
                actiond=agent3.act(state3)
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
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bk[:len(inp)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                    steps_epi.append(steps_ep)
                    if reward==1 or reward1==1:
                        solved+=1
                        cumre+=1
                        reward_episode.append(1)
                        cumulative_reward.append(cumre)
                    else:
                        solved+=0
                        cumre-=1
                        reward_episode.append(0)
                        cumulative_reward.append(cumre)
            total_episodes.append(reward_episode)
    
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
    plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(total_fidelity))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()



state_size=4
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
action_size=len(actions_list)#env.action_space.n





agent = DQNAgent(state_size, action_size)
agent=Dqn([0],agent)
agentO = DQNAgent(state_size, action_size)
agentO=Dqn([[1]],agentO)
onebitsimulation(agent,agentO)

agent = DQNAgent(state_size, action_size)
agent=Dqn([0,0],agent)
agentO = DQNAgent(state_size, action_size)
agentO=Dqn([0,1],agentO)
agentT = DQNAgent(state_size, action_size)
agentT=Dqn([1,0],agentT)
agentTh = DQNAgent(state_size, action_size)
agentTh=Dqn([1,0],agentTh)
twobitsimulation(agent,agentO,agentT,agentTh)




agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQN00.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNN01.h5")
onebitsimulation(agent,agent1)
agent3 = DQNAgent(state_size, action_size)
agent3.load("./QP84DQNN11.h5")
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQNN10.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNN01.h5")
agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQN00.h5")
twobitsimulation(agent,agent1,agent2,agent3)
agent7 = DQNAgent(state_size, action_size)
agent7.load("./QP84DQN111.h5")
agent6 = DQNAgent(state_size, action_size)
agent6.load("./QP84DQN110.h5")
agent5 = DQNAgent(state_size, action_size)
agent5.load("./QP84DQN101.h5")
agent4 = DQNAgent(state_size, action_size)
agent4.load("./QP84DQN100.h5")
agent3 = DQNAgent(state_size, action_size)
agent3.load("./QP84DQNN11.h5")
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQNN10.h5")
agent1 = DQNAgent(state_size, action_size)
agent1.load("./QP84DQNN01.h5")
agent = DQNAgent(state_size, action_size)
agent.load("./QP84DQN00.h5")
threebitsimulation(agent,agent1,agent2,agent3,agent4,agent5,agent6,agent7)
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
agent7.load("./QP84DQN1000.h5")
agent6.load("./QP84DQN1001.h5")
agent5.load("./QP84DQN1010.h5")
agent4.load("./QP84DQN1011.h5")
agent3.load("./QP84DQN1100.h5")
agent2.load("./QP84DQN1101.h5")
agent1.load("./QP84DQN1110.h5")
agent.load("./QP84DQN1.h5")
fourbitsimulation(agent,agent1,agent2,agent3,agent4,agent5,agent6,agent7,agent8,agent9,agent10,agent11,agent12,agent13,agent14,agent15)







































    
#done = False
batch_size=24
EPISODES=100
solved=0
steps_epi=[]
qval=[]
qval_pr=[]
total_episodes=[]
qpt=Qprotocol(4)
r=0
cumulative_reward=[]
for e in range(EPISODES):
        state_n=qpt.reset(4,np.random.randint(0,2,1))
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,dedoce=reset()
        state=state_n
        steps_ep=0
        #state=env.reset()
        reward_episode=[]
        #done=False
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
        state = np.array(state[0])
        state1=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        state2=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state3=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state4=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state5=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state6=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state7=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state8=np.reshape(state, [1, state_size])
        #state9=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state10=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state11=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state12=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state13=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state14=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state15=np.reshape(state, [1, state_size])
        #state = np.array(state[0])
        #state16=np.reshape(state, [1, state_size])
        #while done4!=True or 
        while done1!=True or done2!=True:# or done3!=True or done8!=True or done5!=True or done6!=True or done7!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
        #for time in range(50):
            #actiona=agent.act(state)
            actiona=agent.act(state1)
            actionb=agent1.act(state2)
            #actionc=agent2.act(state3)
            #actiond=agent3.act(state4)            
            #actione=agent4.act(state5)
            #actionf=agent5.act(state6)
            #actiong=agent6.act(state7)
            #actionh=agent7.act(state8)
            #actionj=agent8.act(state9)
            #actionk=agent9.act(state10)
            #actionl=agent10.act(state11)
            #actionu=agent11.act(state12)            
            #actioni=agent12.act(state13)
            #actiono=agent13.act(state14)
            #actionp=agent14.act(state15)
            #actionq=agent15.act(state16)
            #print('This is action b {}'.format(actionb))
            #action=(actiona,actionb)
            actiona=np.array(actiona)
            actionb=np.array(actionb)
            #actionc=np.array(actionc)
            #actiond=np.array(actiond)
            #actione=np.array(actione)
            #actionf=np.array(actionf)
            #actiong=np.array(actiong)
            #actionh=np.array(actionh)
            #actionj=np.array(actiona)
            #actionk=np.array(actionb)
            #actionl=np.array(actionc)
            #actionu=np.array(actiond)
            #actioni=np.array(actione)
            #actiono=np.array(actionf)
            #actionp=np.array(actiong)
            #actionq=np.array(actionh)
            actionZ = actions_list[actiona]
            actionO = actions_list[actionb]
            #actionT = actions_list[actionc]
            #actionTh = actions_list[actiond]
            #actionF = actions_list[actione]
            #actionFi = actions_list[actionf]
            #actionS = actions_list[actiong]
            #actionO = actions_list[actionh]
            #actionZq = actions_list[actiona]
            #actionOq = actions_list[actionb]
            #actionTq = actions_list[actionc]
            #actionThq = actions_list[actiond]
            #actionFq = actions_list[actione]
            #actionFiq = actions_list[actionf]
            #actionSq = actions_list[actiong]
            #actionOq = actions_list[actionh]
            stat1,re1,do1,action_h1=qpt.step(actionZ)
            stat2,re2,do2,action_h2=qpt.step(actionO)
            #stat3,re3,do3,action_h3=qpt.step(actionT)
            #stat4,re4,do4,action_h4=qpt.step(actionTh)
            #stat5,re5,do5,action_h5=qpt.step(actionF)
            #stat6,re6,do6,action_h6=qpt.step(actionFi)
            #stat7,re7,do7,action_h7=qpt.step(actionS)
            #stat8,re8,do8,action_h8=qpt.step(actionO)
            #stat9,re9,do9,action_h9=qpt.step(actionZq)
            #stat10,re10,do10,action_h10=qpt.step(actionOq)
            #stat11,re11,do11,action_h11=qpt.step(actionTq)
            #stat12,re12,do12,action_h12=qpt.step(actionThq)
            #stat13,re13,do13,action_h13=qpt.step(actionFq)
            #stat14,re14,do14,action_h14=qpt.step(actionFiq)
            #stat15,re15,do15,action_h15=qpt.step(actionSq)
            #stat16,re16,do16,action_h16=qpt.step(actionOq)
            #stat1,re1,do1,action_h1,bob_key1=step(actionZ,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done1,dedoce,verbose=0,)
            #stat2,re2,do2,action_h2,bob_key2=step(actionO,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done2,dedoce,verbose=0,)
            #stat3,re3,do3,action_h3,bob_key3=step(actionT,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done3,dedoce,verbose=0,)
            #stat4,re4,do4,action_h4,bob_key4=step(actionTh,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done4,dedoce,verbose=0,)
            #stat5,re5,do5,action_h5,bob_key5=step(actionF,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done5,dedoce,verbose=0,)
            #stat6,re6,do6,action_h6,bob_key6=step(actionFi,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done6,dedoce,verbose=0,)
            #stat7,re7,do7,action_h7,bob_key7=step(actionS,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done7,dedoce,verbose=0,)
            #stat8,re8,do8,action_h8,bob_key8=step(actionO,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done8,dedoce,verbose=0,)
            #stat9,re9,do9,action_h9,bob_key9=step(actionZq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done9,dedoce,verbose=0,)
            #stat10,re10,do10,action_h10,bob_key10=step(actionOq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done10,dedoce,verbose=0,)
            #stat11,re11,do11,action_h11,bob_key11=step(actionTq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done11,dedoce,verbose=0,)
            #stat12,re12,do12,action_h12,bob_key12=step(actionThq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done12,dedoce,verbose=0,)
            #stat13,re13,do13,action_h13,bob_key13=step(actionFq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done13,dedoce,verbose=0,)
            #stat14,re14,do14,action_h14,bob_key14=step(actionFiq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done14,dedoce,verbose=0,)
            #stat15,re15,do15,action_h15,bob_key15=step(actionSq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done15,dedoce,verbose=0,)
            #stat16,re16,do16,action_h16,bob_key16=step(actionOq,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done16,dedoce,verbose=0,)
            #next_state, reward, done, _, info = env.step(action)
            reward1 = re1#ward
            reward2 = re2
            #reward3 = re3#ward
            #reward4 = re4
            #reward5 = re5#ward
            #reward6 = re6
            #reward7 = re7#ward
            #reward8 = re8
            #reward9 = re9
            #reward10 = re10
            #reward11 = re11
            #reward12 = re12
            #reward13 = re13
            #reward14 = re14
            #reward15 = re15
            #reward16 = re16
            done1=do1
            done2=do2
            #done3=do3
            #done4=do4
            #done5=do5
            #done6=do6
            #done7=do7
            #done8=do8
            #done9=do9
            #done10=do10
            #done11=do11
            #done12=do12
            #done13=do13
            #done14=do14
            #done15=do15
            #done16=do16
            steps_ep+=1
            next_state1=np.array(stat1[0])
            print(next_state1)
            next_state1= np.reshape(next_state1, [1, state_size])
            next_state2=np.array(stat2[0])
            print(next_state2)
            next_state2= np.reshape(next_state2, [1, state_size])
            #next_state3=np.array(stat3[0])
            #next_state3= np.reshape(next_state3, [1, state_size])
            #next_state4=np.array(stat4[0])
            #next_state4= np.reshape(next_state4, [1, state_size])
            #next_state5=np.array(stat5[0])
            #next_state5= np.reshape(next_state5, [1, state_size])
            #next_state6=np.array(stat6[0])
            #next_state6= np.reshape(next_state6, [1, state_size])
            #next_state7=np.array(stat7[0])
            #next_state7= np.reshape(next_state7, [1, state_size])
            #next_state8=np.array(stat8[0])
            #next_state8= np.reshape(next_state8, [1, state_size])
            #next_state9=np.array(stat9[0])
            #next_state9= np.reshape(next_state9, [1, state_size])
            #next_state10=np.array(stat10[0])
            #next_state10= np.reshape(next_state10, [1, state_size])
            #next_state11=np.array(stat11[0])
            #next_state11= np.reshape(next_state11, [1, state_size])
            #next_state12=np.array(stat12[0])
            #next_state12= np.reshape(next_state12, [1, state_size])
            #next_state13=np.array(stat13[0])
            #next_state13= np.reshape(next_state13, [1, state_size])
            #next_state14=np.array(stat14[0])
            #next_state14= np.reshape(next_state14, [1, state_size])
            #next_state15=np.array(stat15[0])
            #next_state15= np.reshape(next_state15, [1, state_size])
            #next_state16=np.array(stat16[0])
            #next_state16= np.reshape(next_state16, [1, state_size])
            #next_state= np.reshape(next_state, [1, state_size])
            #agent.memorize(state, actiona, reward, next_state, done)
            state1 = next_state1
            state2 = next_state2
            #state3 = next_state3
            #state4 = next_state4
            #state5 = next_state5
            #state6 = next_state6
            #state7 = next_state7
            #state8 = next_state8
            #state9 = next_state9
            #state10 = next_state10
            #state11 = next_state11
            #state12 = next_state12
            #state13 = next_state13
            #state14 = next_state14
            #state15 = next_state15
            #state16 = next_state16
            #done=do
            if done1 or done2: # or done3 or done4 or done5 or done6 or done7 or done8 or done9 or done10 or done11 or done12 or done13 or done14 or done15 or done16:
                steps_epi.append(steps_ep)
                #print('These are rewards {}'.format(re1,re2,re3,re4,re5,re6,re7,re8))
                if reward1==1 or reward2==1:# or reward3==1 or reward4==1 or reward5==1 or reward6==1 or reward7==1 or reward8==1 or reward9==1 or reward10==1 or reward11==1 or reward12==4 or reward13==1 or reward14==1 or reward15==1 or reward16==1:
                    solved+=1
                    print('The reward is {}'.format(solved))
                    r+=1
                    reward_episode.append(1)
                    cumulative_reward.append(r)
                else:
                    r-=1
                    reward_episode.append(0)
                    cumulative_reward.append(r)
                #print("episode : {}/{},reward {}".format(e, EPISODES,reward))#, solved, agent.epsilon,reward))
                break 
            #print('The agent memory {}'.format(len(agent.memory)))
            #if len(agent.memory) > batch_size:
            #    agent.replay(batch_size)
            #    qval.append(agent.q_value)
            #    qval_pr.append(agent.q_value_pr)
                #print(qval)
            #agent.save("./QP84DQN.h5")
        total_episodes.append(reward_episode)

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
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
#plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()
plt.figure(figsize=(13, 13))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(steps_epi)
plt.xlabel(f'Number of episodes')
plt.ylabel('Number of steps')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()