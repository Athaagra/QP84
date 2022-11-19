#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:51:14 2022

@author: Optimus
"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
#import torch.optim as optim
import gym
import random
import math
import time
import matplotlib.pyplot as plt

#env = gym.make("CartPole-v1")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from ple import PLE
#from ple.games.flappybird import FlappyBird
import sys
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
    #data1 = np.random.randint(0,2,1)
    data1=np.array([1])
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
	




EPISODES=100
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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
            #target_fT=self.modelT.predict(state)
            #print(target_f)
            artatt=np.argmax(target_f)
            #print(artatt)
            self.q_value_pr.append(target_f[0][artatt])
            #print('This is the target {}'.format(target))
            #print('This is the target f {} action {}'.format(target_f[0],action))
            print(action)
            target_f[0][action]=target
            #target_fT[0][action[1]]=target
            history=self.model.fit(state, target_f, epochs=1,verbose=0,batch_size=batch_size,callbacks=[callback])
            #history=self.modelT.fit(state, target_fT, epochs=1,verbose=0,batch_size=batch_size,callbacks=[callback])
            #print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

if __name__=="__main__":
    #env=gym.make('CartPole-v1')
    state_size=4#env.observation_space.shape[0]
    #action_size=#env.action_space.n
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    action_size=len(actions_list)#env.action_space.n
    print(action_size)
    agent = DQNAgent(state_size, action_size)
    #agent.load("./QP84DQN.h5")
    done = False
    batch_size=32
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    r=0
    cumulative_reward=[]
    for e in range(EPISODES):
        state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        state=state_n
        steps_ep=0
        #state=env.reset()
        reward_episode=[]
        state = np.array(state[0])
        state=np.reshape(state, [1, state_size])
        while done!= True:
        #for time in range(50):
            #actiona=agent.act(state)
            actiona=agent.act(state)
            #print('This is action b {}'.format(actionb))
            #action=(actiona,actionb)
            actiona=np.array(actiona)
            action = actions_list[actiona]
            stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
            #next_state, reward, done, _, info = env.step(action)
            reward = re#ward
            steps_ep+=1
            next_state=np.array(stat[0])
            next_state= np.reshape(next_state, [1, state_size])
            #next_state= np.reshape(next_state, [1, state_size])
            agent.memorize(state, actiona, reward, next_state, done)
            state = next_state
            reward_episode.append(re)
            done=do
            if done:
                steps_epi.append(steps_ep)
                if reward==1:
                    solved+=1                
                print("episode : {}/{},reward {}".format(e, EPISODES,reward))#, solved, agent.epsilon,reward))
                break 
            #print('The agent memory {}'.format(len(agent.memory)))
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                qval.append(agent.q_value)
                qval_pr.append(agent.q_value_pr)
                #print(qval)
            agent.save("./QP84DQNforOne.h5")
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
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
#plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()

plt.figure(figsize=(13, 13))
#print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
plt.plot(qval[0])
plt.plot(qval_pr[0])
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q value')
plt.title('The Q value')#.format(solved/EPISODES))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()

#env=gym.make('CartPole-v1')
state_size=4#env.observation_space.shape[0]
#action_size=#env.action_space.n
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
action_size=len(actions_list)#env.action_space.n
agent2 = DQNAgent(state_size, action_size)
agent2.load("./QP84DQN.h5")
done = False
batch_size=24
solved=0
steps_epi=[]
qval=[]
qval_pr=[]
total_episodes=[]
r=0
cumulative_reward=[]
for e in range(EPISODES):
        state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        state=state_n
        steps_ep=0
        #state=env.reset()
        reward_episode=[]
        state = np.array(state[0])
        state=np.reshape(state, [1, state_size])
        while done!= True:
        #for time in range(50):
            #actiona=agent.act(state)
            actiona=agent.act(state)
            actionb=agent2.act(state)
            #print('This is action b {}'.format(actionb))
            #action=(actiona,actionb)
            actiona=np.array(actiona)
            actionb=np.array(actionb)
            actionZ = actions_list[actiona]
            actionO = actions_list[actionb]
            stat1,re1,do1,action_h1,bob_key1=step(actionZ,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done,verbose=0,)
            stat2,re2,do2,action_h2,bob_key2=step(actionO,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done,verbose=0,)
            #next_state, reward, done, _, info = env.step(action)
            reward1 = re1#ward
            reward2 = re2
            done1=do1
            done2=do2
            steps_ep+=1
            next_state=np.array(stat[0])
            next_state= np.reshape(next_state, [1, state_size])
            #next_state= np.reshape(next_state, [1, state_size])
            #agent.memorize(state, actiona, reward, next_state, done)
            state = next_state
            reward_episode.append(re)
            #done=do
            if done1 or done2:
                steps_epi.append(steps_ep)
                if reward==1:
                    solved+=1                
                print("episode : {}/{},reward {}".format(e, EPISODES,reward))#, solved, agent.epsilon,reward))
                break 
            #print('The agent memory {}'.format(len(agent.memory)))
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                qval.append(agent.q_value)
                qval_pr.append(agent.q_value_pr)
                #print(qval)
            #agent.save("./QP84DQN.h5")
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
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(cumulative_reward)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
#plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()

plt.figure(figsize=(13, 13))
#print('The simulation has been solved the environment DQN:{}'.format(solved/EPISODES))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_epi))))
plt.plot(qval[0])
plt.plot(qval_pr[0])
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q value')
plt.title('The Q value')#.format(solved/EPISODES))
plt.grid(True,which="both",ls="--",c='gray')
plt.show()