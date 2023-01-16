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
    def reset(self,inputd,encode=encoded,decode=decoded):
        import numpy as np
        self.max_moves = 4
        # State for alice
        #self.data0=np.random.randint(0,2,2)
        self.data1=np.array(inputd)#[0,0,1]#np.random.randint(0,2,1)#[1]
        print('This is data1 {}'.format(type(self.data1)))
        #self.data1 = np.random.randint(0,2,1)
        #self.data2 = np.random.randint(0,2,2)
        self.data0=encode(self.data1,len(self.data1))
        #print(self.data0)
        self.data2=decode(self.data0,len(self.data0))
        print('This is data2 {}'.format(type(self.data2)))
        #if self.data1==self.data2:
        #    print('True')
        z=[self.data1[i]==self.data2[i] for i in range(len(self.data1))]
        z=np.array(z)
        if z.all():
            print('True')
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
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q1=np.zeros(q)
total_episodes=[]
env=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env.reset([0,0,0,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q1[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q1[q_val]=(1-learning_rate)*Q1[q_val]+learning_rate * (re + gamma * max(Q1[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q2=np.zeros(q)
total_episodes=[]
env1=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env1.reset([0,0,0,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q2[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env1.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q2[q_val]=(1-learning_rate)*Q2[q_val]+learning_rate * (re + gamma * max(Q2[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()



import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q3=np.zeros(q)
total_episodes=[]
env3=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env3.reset([0,0,1,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q3[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env3.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q3[q_val]=(1-learning_rate)*Q3[q_val]+learning_rate * (re + gamma * max(Q3[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q4=np.zeros(q)
total_episodes=[]
env4=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env4.reset([0,0,1,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q4[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env4.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q4[q_val]=(1-learning_rate)*Q4[q_val]+learning_rate * (re + gamma * max(Q4[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q5=np.zeros(q)
total_episodes=[]
env5=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env5.reset([0,1,0,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q5[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env5.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q5[q_val]=(1-learning_rate)*Q5[q_val]+learning_rate * (re + gamma * max(Q5[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q6=np.zeros(q)
total_episodes=[]
env6=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env6.reset([0,1,0,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q6[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env6.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q6[q_val]=(1-learning_rate)*Q6[q_val]+learning_rate * (re + gamma * max(Q6[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q7=np.zeros(q)
total_episodes=[]
env7=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env7.reset([0,1,1,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q7[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env7.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q7[q_val]=(1-learning_rate)*Q7[q_val]+learning_rate * (re + gamma * max(Q7[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q8=np.zeros(q)
total_episodes=[]
env8=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env8.reset([0,1,1,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q8[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env8.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q8[q_val]=(1-learning_rate)*Q8[q_val]+learning_rate * (re + gamma * max(Q8[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q9=np.zeros(q)
total_episodes=[]
env9=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env.reset([1,0,0,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q9[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q9[q_val]=(1-learning_rate)*Q9[q_val]+learning_rate * (re + gamma * max(Q9[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q10=np.zeros(q)
total_episodes=[]
env10=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env1.reset([1,0,0,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q10[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env10.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q10[q_val]=(1-learning_rate)*Q10[q_val]+learning_rate * (re + gamma * max(Q10[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()



import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q11=np.zeros(q)
total_episodes=[]
env11=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env11.reset([1,0,1,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q11[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env11.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q11[q_val]=(1-learning_rate)*Q11[q_val]+learning_rate * (re + gamma * max(Q11[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q12=np.zeros(q)
total_episodes=[]
env12=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env4.reset([1,0,1,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q12[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env12.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q12[q_val]=(1-learning_rate)*Q12[q_val]+learning_rate * (re + gamma * max(Q12[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q13=np.zeros(q)
total_episodes=[]
env13=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env5.reset([1,1,0,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q13[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env13.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q13[q_val]=(1-learning_rate)*Q13[q_val]+learning_rate * (re + gamma * max(Q13[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q14=np.zeros(q)
total_episodes=[]
env14=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env14.reset([1,1,0,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q14[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env14.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q14[q_val]=(1-learning_rate)*Q14[q_val]+learning_rate * (re + gamma * max(Q14[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()


import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q15=np.zeros(q)
total_episodes=[]
env15=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env15.reset([1,1,1,0])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q15[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env15.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q15[q_val]=(1-learning_rate)*Q15[q_val]+learning_rate * (re + gamma * max(Q15[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

import numpy as np
episodes=1000
solved=0
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
steps_ep=[]
q=(4,len(actions_list))
#print(state_n)
Q16=np.zeros(q)
total_episodes=[]
env16=Qprotocol(4)
cum=[]
cumre=0
re=0
error=0
testing=[]
for episode in range(episodes):
    import matplotlib.pyplot as plt
    gamma= 0.01
    learning_rate=0.001
    state_n=env16.reset([1,1,1,1])
    error+=1
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        random_values=Q16[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        q_val=(int(state_n[0][0]),actiona)
        action=np.array(actions_list[actiona])
        testing.append([action,re])
        #print('This is the action {}'.format(action))
        stat,re,do,action_h=env16.step(action)
        #print('the reward is {},the is done {}, this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        Q16[q_val]=(1-learning_rate)*Q16[q_val]+learning_rate * (re + gamma * max(Q16[int(stat[0][0])]))
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        print(re)
        state_n=stat
    if do==True:
        cumre+=re
        cum.append(cumre)
        if reward_episode[-1]==1:
            solved+=1
            steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_episodes)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()
#print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
#print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of episode')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()

total_re=[]
solved=0
steps_ep=[]
episodes = 100
envs=Qprotocol(4)
for _ in range(episodes):
    state_n=envs.reset(np.random.randint(0,2,4))
#    epochs, penalties, reward = 0, 0, 0
    state_n1=state_n[0][0]
    print(state_n1)
    done1 = False
    done2 = False
    done3 = False
    done4 = False
    done5 = False
    done6 = False
    done7 = False
    done8 = False
    done9 = False
    done10 = False
    done11 = False
    done12 = False
    done13 = False
    done14 = False
    done15 = False
    done16 = False
    steps=0
    while done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
        action1 = np.argmax(Q1[int(state_n1)])
        action2 = np.argmax(Q2[int(state_n1)])
        action3 = np.argmax(Q3[int(state_n1)])
        action4 = np.argmax(Q4[int(state_n1)])
        action5 = np.argmax(Q5[int(state_n1)])
        action6 = np.argmax(Q6[int(state_n1)])
        action7 = np.argmax(Q7[int(state_n1)])
        action8 = np.argmax(Q8[int(state_n1)])
        action9 = np.argmax(Q9[int(state_n1)])
        action10 = np.argmax(Q10[int(state_n1)])
        action11 = np.argmax(Q11[int(state_n1)])
        action12 = np.argmax(Q12[int(state_n1)])
        action13 = np.argmax(Q13[int(state_n1)])
        action14 = np.argmax(Q14[int(state_n1)])
        action15 = np.argmax(Q15[int(state_n1)])
        action16 = np.argmax(Q16[int(state_n1)])
        #print(action)
        action01=np.array(actions_list[action1])
        action02=np.array(actions_list[action2])
        action03=np.array(actions_list[action3])
        action04=np.array(actions_list[action4])
        action05=np.array(actions_list[action5])
        action06=np.array(actions_list[action6])
        action07=np.array(actions_list[action7])
        action08=np.array(actions_list[action8])
        action09=np.array(actions_list[action9])
        action10=np.array(actions_list[action10])
        action11=np.array(actions_list[action11])
        action12=np.array(actions_list[action12])
        action13=np.array(actions_list[action13])
        action14=np.array(actions_list[action14])
        action15=np.array(actions_list[action15])
        action16=np.array(actions_list[action16])
        stat1,re1,do1,action_h1=envs.step(action01)
        stat2,re2,do2,action_h2=envs.step(action02)
        stat3,re3,do3,action_h3=envs.step(action03)
        stat4,re4,do4,action_h4=envs.step(action04)
        stat5,re5,do5,action_h5=envs.step(action05)
        stat6,re6,do6,action_h6=envs.step(action06)
        stat7,re7,do7,action_h7=envs.step(action07)
        stat8,re8,do8,action_h8=envs.step(action08)
        stat9,re9,do9,action_h9=envs.step(action09)
        stat10,re10,do10,action_h10=envs.step(action10)
        stat11,re11,do11,action_h11=envs.step(action11)
        stat12,re12,do12,action_h12=envs.step(action12)
        stat13,re13,do13,action_h13=envs.step(action13)
        stat14,re14,do14,action_h14=envs.step(action14)
        stat15,re15,do15,action_h15=envs.step(action15)
        stat16,re16,do16,action_h16=envs.step(action16)
        done1=do1
        done2=do2
        done3=do3
        done4=do4
        done5=do5
        done6=do6
        done7=do7
        done8=do8
        done9=do9
        done10=do10
        done11=do11
        done12=do12
        done13=do13
        done14=do14
        done15=do15
        done16=do16
        steps+=1
        reward1=re1
        reward2=re2
        reward3=re3
        reward4=re4
        reward5=re5
        reward6=re6
        reward7=re7
        reward8=re8
        reward9=re9
        reward10=re10
        reward11=re11
        reward12=re12
        reward13=re13
        reward14=re14
        reward15=re15
        reward16=re16
        state_n1=stat1[0][0]
        state_n1=stat2[0][0]
        state_n1=stat3[0][0]
        state_n1=stat4[0][0]
        state_n1=stat5[0][0]
        state_n1=stat6[0][0]
        state_n1=stat7[0][0]
        state_n1=stat8[0][0]
        state_n1=stat9[0][0]
        state_n1=stat10[0][0]
        state_n1=stat11[0][0]
        state_n1=stat12[0][0]
        state_n1=stat13[0][0]
        state_n1=stat14[0][0]
        state_n1=stat15[0][0]
        state_n1=stat16[0][0]
        #print(re)
        if do1 ==True or do2 == True or do3==True or do4==True or do5==True or do6==True or do7==True or do8==True or do9 ==True or do10 == True or do11==True or do12==True or do13==True or do14==True or do15==True or do16==True:
            steps_ep.append(steps)
            if reward1==1 or reward2==1 or reward3==1 or reward4==1 or  reward5==1 or reward6==1 or  reward7==1 or reward8==1 or reward9==1 or reward10==1 or reward11==1 or reward12==1 or  reward13==1 or reward14==1 or  reward15==1 or reward16==1:
                total_re.append(1)
                solved+=1
            else:
                total_re.append(0)
                solved+=0
            #if re==1:
from scipy.stats import mannwhitneyu
# seed the random number generator
resultss=[]
if sum(total_re)!=sum(envs.error_counter):
    stat, pvalue = mannwhitneyu(total_re, envs.error_counter)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    if pvalue > 0.05:
        print('We accept the null hypothesis')
        resultss.append(['Evolutionary Strategy Reward We accept the null hypothesis:',pvalue])
    else:
        print("The p-value is less than we reject the null hypothesis")
        resultss.append(['Evolutionary Strategy The p-value is less than we reject the null hypothesis:',pvalue])
else:
    print('continue')
print('The simulation has been solved the environment Q learning test set:{}'.format(solved/episodes))
print('The number of steps per episode that solved test set:{}'.format(np.round(np.mean(steps_ep))))
plt.figure(figsize=(13, 13))
plt.plot(total_re)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Q learning test set:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_ep)
plt.xlabel(f'Number of episode')
plt.ylabel('Number of Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum)
plt.xlabel(f'Number of steps')
plt.ylabel('Cumulative Rewards')
plt.grid(True,which="both",ls="--",c='gray')
#plt.title('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
plt.show()