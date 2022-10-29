#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:26:21 2022

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
	
Rewards accumulate: negative points for wrong guess, 
positive points for correct guess
Game terminates with correct key or N moves

"""
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import random 
import math 
import time
from torch.autograd import Variable
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
            print("Alice has added to the counter a variable")
	
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
            reward = 0
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
            reward = 0
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
    data1 = np.random.randint(0,2,1)
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



#if gpu is to be used 
use_cuda=torch.cuda.is_available()

device=torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor
gamma= 0.99
seed_value=0
torch.manual_seed(seed_value)
random.seed(seed_value)
#learning_rate=1
learning_rate=0.03
number_of_inputs = 4
number_of_outputs = 12
replay_mem_size=30
egreedy=1
egreedy_final=0.89
egreedy_decay=500
update_target_frequency = 20
batch_size=10
hidden_layer=64
steps=[]
clip_error=False

def calculate_epsilon(steps_done):
    epsilon=egreedy_final + (egreedy - egreedy_final) * \
        math.exp(-1. * steps_done / egreedy_decay)
    return epsilon

class ExperienceReplay(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
    
    def push(self, state, action, new_state, reward, done):
        transition=(state, action, new_state, reward, done)
        if self.position >=len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position]=transition
        self.position= (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return zip(*random.sample(self.memory,batch_size))
    def __len__(self):
        return len(self.memory)
class NeuralNetwork(nn.Module):
    #def __init__(self):
    #    super(NeuralNetwork, self).__init__()
    #    self.linear1 = nn.Linear(number_of_inputs,number_of_outputs)
        
    #def forward(self, x):
    #    output = self.linear1(x)
    #    return output
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        self.linear2 = nn.Linear(hidden_layer,number_of_outputs)
        self.activation = nn.Tanh()
    def forward(self, x):
        output = self.linear1(x)
        output = self.activation(output)
        output1 = self.linear2(output)
        return output1

class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss() 
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        self.update_target_counter = 0
    def select_action(self,state,epsilon):
         random_for_egreedy = torch.rand(1)[0]
         if random_for_egreedy < egreedy:
             with torch.no_grad():
                 state=Tensor(state).to(device)
                 action_from_nn=self.nn(state)
                 action=torch.max(action_from_nn,0)[1]
                 actiona=action.item()
                 #print('This is torch no grad {}'.format(actiona))
                 #action=np.array(actions_list[actiona])
                 
         else:
            random_values=Q[int(state_n[0][0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            actiona=np.argmax(random_values)
            #print('Select a random action {}'.format(actiona))
            #action=np.array(actions_list[actiona])
         return actiona      
    def optimize(self):
        if (len(memory) < batch_size):
            return
        state,action,new_state,reward,done=memory.sample(batch_size)
        print(state,action,new_state,reward,done)
        state=Tensor(state).to(device)
        new_state=Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)
        new_state_values=self.target_nn(new_state).detach()
        max_new_state_values = torch.max(new_state_values, 1)[0]
        target_value=reward + gamma * max_new_state_values
        #target_value=reward + (1- done) * gamma * max_new_state_values
        print('This is the state {}'.format(self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)))
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        print('This is the target value {}'.format(predicted_value))          
        loss = self.loss_func(predicted_value, target_value)
            
        self.optimizer.zero_grad()
        loss.backward()
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp(-1,1)
        self.optimizer.step()
            
        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
            
        self.update_target_counter +=1
            
memory=ExperienceReplay(replay_mem_size)          
episodes=100
cumulative_reward=[]
solved=0
r=0
frames_total=0
steps_ep=[]
total_episodes=[]
qnet_agent=QNet_Agent()
for episode in range(episodes):
    import numpy as np
    import matplotlib.pyplot as plt
    #def __init__(self):
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    state_n,action,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
    q=(4,len(actions_list))
    Q=np.zeros(q)
    do=False
    reward_episode=[]
    steps=0
    while do!=True:
        steps+=1
        frames_total+=1
        epsilon=calculate_epsilon(frames_total)
        action_index=qnet_agent.select_action(state_n[0],epsilon)
        action=np.array(actions_list[action_index])
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        #print('This is the action {}'.format(action)) this is the key of bob {}, this is the bitstring {}'.format(re,do,bob_key,data))
        memory.push(stat[0],action_index,state_n[0],re,do)
        qnet_agent.optimize()
        #print('This is the tabular {}'.format(Q))
        reward_episode.append(re)
        #print(Q)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
    r+=reward_episode[-1]
    cumulative_reward.append(r)
print('The simulation has been solved the environment DQN:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
plt.plot(total_episodes)
plt.title('The simulation has been solved the environment Deep Q learning:{}'.format(solved/episodes))
plt.show()
plt.plot(cumulative_reward)
plt.show()