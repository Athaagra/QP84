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
    def reset(self,encode=encoded,decode=decoded):
        import numpy as np
        self.max_moves = 4
        # State for alice
        #self.data0=np.random.randint(0,2,2)
        self.data1 = np.random.randint(0,2,4)
        #self.data2 = np.random.randint(0,2,2)
        self.data0=encode(self.data1,len(self.data1))
        #print(self.data0)
        self.data2=decode(self.data0,len(self.data0))
        print(self.data2)
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
# =============================================================================
# Q-learning deterministic
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt	
resultss=[]
episodes=100
qp=Qprotocol(4)
solved=0
steps_ep=[]
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]   
q=(6,len(actions_list))
Q=np.zeros(q)
error=0
total_episodes=[]
q_value=[]
for episode in range(episodes):
    #np.random.seed(0)
    gamma= 1
    state=qp.reset()
    gamma= 0.01
    learning_rate=0.001
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
        value=(1-learning_rate)*Q[q_val]+learning_rate * (reward + gamma * max(Q[int(new_state[0][0])]))
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
    else:
        solved+=0 
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])


total_re=[]
solved=0
steps_ep=[]
episodes = 100
for _ in range(episodes):
    state=qp.reset()
#    epochs, penalties, reward = 0, 0, 0
    state=state[0][0]
    #print(state_n1)
    done = False
    steps=0
    while done!=True:# or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True:
        action = np.argmax(Q[int(state)])
        #print(action)
        action=np.array(actions_list[action])
        new_state, reward, done,info=qp.step(action)
        done=done
        steps+=1
        reward=reward
        state=new_state[0][0]
        #print(re)
        if done ==True:# or do2 == True or do3==True or do4==True or do5==True or do6==True or do7==True or do8==True:
            steps_ep.append(steps)
            if reward==1:# or reward2==1 or reward3==1 or reward4==1 or  reward5==1 or reward6==1 or  reward7==1 or reward8==1:
                total_re.append(1)
                solved+=1
            else:
                total_re.append(0)
                solved+=0
            #if re==1:
error=np.array(qp.error_counter)[100:-1]
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment Q-learning Equation:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Q-learning Equation:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(q_value,c='orange')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The q-value during the training')
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_ep,c='grey')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The steps during the training:{}'.format(np.mean(steps_ep)))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(error,c='red')
plt.xlabel(f'Number of episodes')
plt.ylabel('Error')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('Number of errors:{}'.format(sum(error)))
plt.show()
resultss.append(['Qlearning Reward :',solved/episodes])
resultss.append(['Qlearning Steps Reward :',np.mean(steps_ep)])
resultss.append(['Qlearning Error Reward :',sum(error)])
from scipy.stats import mannwhitneyu
# seed the random number generator
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
    
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import gym
import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys



EPISODES=80
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
        self.learning_rate = 1e-4#20.25##
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
            history=self.model.fit(state, target_f, epochs=1,verbose=0,batch_size=batch_size)
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
    qpDqn=Qprotocol(4)
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
        state=qpDqn.reset()
        state=state
        steps_ep=0
        done=False
        reward_episode=[]
        state = np.array(state[0])
        state=np.reshape(state, [1, state_size])
        while done != True:
        #for time in range(50):
            #actiona=agent.act(state)
            actiona=agent.act(state)
            #print('This is action b {}'.format(actionb))
            #action=(actiona,actionb)
            actiona=np.array(actiona)
            action = actions_list[actiona]
            new_state, reward, done,info=qpDqn.step(action)
            steps_ep+=1
            next_state=np.array(new_state[0])
            next_state= np.reshape(next_state, [1, state_size])
            agent.memorize(state, actiona, reward, next_state, done)
            state = next_state
            reward_episode.append(reward)
            print(done)
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
            agent.save("./QP84DQN4QuantumChannel.h5")
        print(reward_episode,e)
        if len(reward_episode)!=0:
            r+=reward_episode[-1]
            cumulative_reward.append(r)
            total_episodes.append(reward_episode[-1])
        else:
            print(reward_episode,e)

#env=gym.make('CartPole-v1')
state_size=4#env.observation_space.shape[0]
#action_size=#env.action_space.n
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
action_size=len(actions_list)#env.action_space.n
agent = DQNAgent(state_size, action_size)
#agent.load("./QP84DQN1QuantumChannel.h5")
done = False
solved=0
steps_epi=[]
batch_size=32
qval=[]
qval_pr=[]
total_episodes=[]
r=0
cumulative_reward=[]
for e in range(EPISODES):
        state=qpDqn.reset()
        state=state
        steps_ep=0
        #state=env.reset()
        done=False
        reward_episode=[]
        state = np.array(state[0])
        state=np.reshape(state, [1, state_size])
        while done!= True:
            actiona=agent.act(state)
            actiona=np.array(actiona)
            action= actions_list[actiona]
            new_state, reward, done,info=qpDqn.step(action)
            reward = reward
            steps_ep+=1
            next_state=np.array(state[0])
            next_state= np.reshape(next_state, [1, state_size])
            state = next_state
            reward_episode.append(reward)
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
            #agent.save("./QP84DQN.h5")
        r+=reward_episode[-1]
        cumulative_reward.append(r)
        total_episodes.append(reward_episode[-1])
error=np.array(qpDqn.error_counter)[100:-1]
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment DQN Equation:{}'.format(solved/EPISODES))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment DQN Equation:{}'.format(solved/EPISODES))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(qval,c='orange')
plt.plot(qval_pr,c='grey')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The q-value during the training')
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_epi,c='grey')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The steps during the training:{}'.format(np.mean(steps_ep)))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(error,c='red')
plt.xlabel(f'Number of episodes')
plt.ylabel('Error')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('Number of errors:{}'.format(sum(error)))
plt.show()

resultss.append(['DQN Reward :',solved/EPISODES])
resultss.append(['DQN Steps Reward :',np.mean(steps_ep)])
resultss.append(['DQN Error Reward :',sum(error)])
from scipy.stats import mannwhitneyu
# seed the random number generator
if sum(total_episodes)!=sum(error):
    stat, pvalue = mannwhitneyu(total_episodes, error)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    # interpret
    
    if pvalue > 0.05:
        print('We accept the null hypothesis')
        resultss.append(['DQN p-value We accept the null hypothesis:',pvalue])
    else:
        print("The p-value is less than we reject the null hypothesis")
        resultss.append(['DQN p-value The p-value is less than we reject the null hypothesis:',pvalue])
else:
    print('continue')

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
        m = R.mean()
        s = R.std()+0.001
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
    model = ANN(D, M, K)
    model.set_params(params)
#     # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    counterr=0
    state=qpEs.reset()
    obs = state[0]#np.concatenate((, state_n[1]), axis=None)#state_n#obs[0]
    obs_dim= len(obs)
    if HISTORY_LENGTH >1:
        state =np.zeros(HISTORY_LENGTH*obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
        actiona = model.sample_action(state)
        action=np.array(actions_list[actiona])
        new_state, reward, done,info=qpEs.step(action)
        obs=new_state[0]
        done=done
        episode_reward += reward
        episode_length +=1
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona,episode_length
#     
if __name__=='__main__':
    model = ANN(D,M,K)
    qpEs=Qprotocol(4)
    if len(sys.argv) > 1 and sys.argv[1] =='play':
        #play with a saved model
        j = np.load('es_qkprotocol_results0000.npz')
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
            num_iters=400,
        )
        model.set_params(best_params)
        total_episodes0=[]
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
            total_episodes0.append(Rew0)
            Rewa += total_episodes0[-1]
            steps_ep.append(steps0)
            if Rew0>0:
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re1.append(cumre)
            else:
                total_ep.append(0)
error=np.array(qpEs.error_counter)[100:-1]
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment Evolutionary Strategy Equation:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_ep)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Evolutionary Strategy Equation:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cum_re1,c='orange')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The q-value during the training')
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_ep,c='grey')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The steps during the training:{}'.format(np.mean(steps_ep)))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(error,c='red')
plt.xlabel(f'Number of episodes')
plt.ylabel('Error')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('Number of errors:{}'.format(sum(error)))
plt.show()

# interpret
resultss.append(['Evolutionary Strategy Reward :',solved/episodes])
resultss.append(['Evolutionary Strategy Steps Reward :',np.mean(steps_ep)])
resultss.append(['Evolutionary Strategy Error Reward :',sum(error)])
from scipy.stats import mannwhitneyu
# seed the random number generator
if sum(total_ep)!=sum(error):
    stat, pvalue = mannwhitneyu(total_ep, error)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    if pvalue > 0.05:
        print('We accept the null hypothesis')
        resultss.append(['Evolutionary Strategy Reward We accept the null hypothesis:',pvalue])
    else:
        print("The p-value is less than we reject the null hypothesis")
        resultss.append(['Evolutionary Strategy The p-value is less than we reject the null hypothesis:',pvalue])
else:
    print('continue')


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym 
import scipy.signal 
import time 

def discounted_cumulative_sums(x, discount):
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
policy_learning_rate = 0.14#3e-4
value_function_learning_rate = 0.14#1e-3
train_policy_iterations = 5000
train_value_iterations = 5000
lam = 0.97
target_kl = 0.01
hidden_sizes = (16, 16)

# True if you want to render the environment
render = False

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
qpPpo=Qprotocol(4)
state=qpPpo.reset()
observation=np.array(state[0])
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
    while done != True:
        if render:
            print(qpPpo.render())
        if len(observation)==2:
            observation=observation[0]
        observation = observation.reshape(1, -1)
        logits, actiona = sample_action(observation)
        log=np.array(logits[0])
        action_actor.append(log[actiona])
        actiona=actiona[0]
        action=np.array(actions_list[actiona])
        new_state, reward, done,info=qpPpo.step(action)
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
            state=qpPpo.reset()
            
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
def save_weights():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    actor.save(path+ '_actorOneQchannel2bit.h5')
    critic.save(path+ '_criticOneQchannel2bit.h5')
#def load_weights():
#    path= '/home/Optimus/Desktop/QuantumComputingThesis/FourDigits/'
#    critic.load_weights(path+ '_criticFourdigits.h5')
#    actor.load_weights(path+ '_actorFourdigits.h5')
#save_weights()
count=0
for i in rewards_during_training:
    if i==1.0:
        count+=1
#save_weights()
#load_weights()
total_episodes=[]
solved=0
episodes=100
steps_ep=[]
r=0
cumulative_rewards=[]
# run infinitely many episodes
for i_episode in range(episodes):
    # reset environment and episode reward
    state=qpPpo.reset()
    observation=state
    ep_reward = 0
    done=False
    steps=0
    while done!=True:
        if len(observation)==2:
            observation=observation[0]
        observation = observation.reshape(1, -1)
        logits, actiona = sample_action(observation)
        actiona=actiona[0]
        steps+=1
        action=np.array(actions_list[actiona])
        new_state, reward, done,info=qpPpo.step(action)
        print('This is the reward {}'.format(reward))
        episode_return += reward
        observation=new_state
        episode_length += 1
        if done==True:
            steps_ep.append(steps)
            total_episodes.append(reward)
            r+=reward
            cumulative_rewards.append(r)
        if reward==1:
            solved+=1
error=np.array(qpPpo.error_counter)[100:-1]
plt.figure(figsize=(13, 13))
print('The simulation has been solved the environment Proximal Policy Optimization Equation:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.xlabel(f'Number of episodes')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Optimization:{}'.format(solved/episodes))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(cumulative_rewards,c='orange')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The cumulative reward')
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(steps_ep,c='grey')
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Steps')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The steps during the training:{}'.format(np.mean(steps_ep)))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(error,c='red')
plt.xlabel(f'Number of episodes')
plt.ylabel('Error')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('Number of errors:{}'.format(sum(error)))
plt.show()

plt.figure(figsize=(13, 13))
plt.plot(q_value_critic)
plt.plot(action_actor)
plt.xlabel(f'Number of Steps of episode')
plt.ylabel('Q-value')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('The simulation has been solved the environment Proximal Policy Evaluation')
plt.show()
resultss.append(['Proximal Policy Reward :',solved/episodes])
resultss.append(['Proximal Policy Steps Reward :',np.mean(steps_ep)])
resultss.append(['Proxumal Policy Error Reward :',sum(error)])
from scipy.stats import mannwhitneyu
# seed the random number generator
if sum(total_episodes) != sum(error):
    stat, pvalue = mannwhitneyu(total_episodes, error)
    print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
    # interpret
    
    if pvalue > 0.05:
        print('We accept the null hypothesis')
        resultss.append(['Proximal Policy Reward We accept the null hypothesis:',pvalue])
    else:
        print("The p-value is less than we reject the null hypothesis")
        resultss.append(['Proximal Policy Reward The p-value is less than we reject the null hypothesis:',pvalue])
else:
    print('identical')
