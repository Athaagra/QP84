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
#def render(alice_datalog,bob_datalog,bob_has_mail):
#    print("---Alice---")
#    print("- Datalog: ", alice_datalog)
#    print("---Bob---")
#    print("- Has Mail: ", bob_has_mail)
#    print("- Datalog: ", bob_datalog)
#    print("")
#    return
import numpy as np   

def step(action,action_history,max_moves, alice_data_counter,data1,alice_datalog,bob_data_counter,alice_observation,bob_key,bob_datalog,cumulative_reward,bob_mailbox,bob_has_mail,done, verbose=0,):
    # Keep track of action list
    action_history.append(action)
    print('This is alice observation {}'.format(alice_observation))
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
            #print('This is the alice datalog {}'.format(alice_datalog))
            alice_data_counter += 1
            #print("Alice has added to the counter a variable")
	
    #if verbose:
    #    print("Alice added data1 to the datalog ", alice_datalog)
	
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
    #print('This Bob datalog {}'.format(bob_datalog))
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
    #alice_observation[(len(action_history)-1)%len(alice_observation)] = action[0]
    #bob_observation = np.concatenate(([bob_has_mail], bob_datalog))
    #state = (alice_observation, bob_observation)
    #minus=1
    # Update the actions that alice and bob took
    #print('This is the alice observation {}'.format(alice_observation))
    #alice_observation[(len(action_history)-1)%len(alice_observation)] = action[0]
    ah=len(action_history)
    print('This is the action history {}'.format(ah))
    if ah >= 6:
        ah=6
    #alice_observation[ah] = action_alice
    bob_observation = np.concatenate(([bob_has_mail], bob_datalog), axis=None)
    state = np.concatenate((alice_observation, bob_observation), axis=None)
    state[ah-1] = ah
    #print('This is the new state {}'.format(state))
    #render(alice_observation,bob_datalog, bob_has_mail)
	
    return state, reward, done, {'action_history':action_history},bob_key


def step2(action,action_history,max_moves, alice_data_counter,data1,alice_datalog,bob_data_counter,alice_observation,bob_key,bob_datalog,cumulative_reward,bob_mailbox,bob_has_mail,done, verbose=0,):
    # Keep track of action list
    action_history.append(action)
    print('This is alice observation {}'.format(alice_observation))
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
            #print('This is the alice datalog {}'.format(alice_datalog))
            alice_data_counter += 1
            #print("Alice has added to the counter a variable")
	
    #if verbose:
    #    print("Alice added data1 to the datalog ", alice_datalog)
	
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
    #print('This Bob datalog {}'.format(bob_datalog))
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
    #alice_observation[(len(action_history)-1)%len(alice_observation)] = action[0]
    #bob_observation = np.concatenate(([bob_has_mail], bob_datalog))
    #state = (alice_observation, bob_observation)
    #minus=1
    # Update the actions that alice and bob took
    #print('This is the alice observation {}'.format(alice_observation))
    #alice_observation[(len(action_history)-1)%len(alice_observation)] = action[0]
    ah=len(action_history)
    print('This is the action history {}'.format(ah))
    if ah >= 6:
        ah=6
    #alice_observation[ah] = action_alice
    bob_observation = np.concatenate(([bob_has_mail], bob_datalog), axis=None)
    state = np.concatenate((alice_observation, bob_observation), axis=None)
    state[ah-1] = ah
    #print('This is the new state {}'.format(state))
    #render(alice_observation,bob_datalog, bob_has_mail)
	
    return state, reward, done, {'action_history':action_history},bob_key

	
def reset(incr):
    max_moves = 4
	
    # State for alice
# =============================================================================
#     data = np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])#np.random.randint(0,2,1)
#     if incr >= 100 and incr < 200:
#         data0=[data[incr-100]]
#     elif incr >= 200 and incr <300:
#         data0=[data[incr-200]]
#     elif incr >= 300 and incr <400:
#         data0=[data[incr-300]]
#     elif incr >= 400 and incr <500:
#         data0=[data[incr-400]]
#     elif incr >= 500 and incr <600:
#         data0=[data[incr-500]]
#     elif incr >= 600 and incr <700:
#         data0=[data[incr-600]]
#     elif incr >= 700 and incr <800:
#         data0=[data[incr-700]]
#     elif incr >= 800 and incr <900:
#         data0=[data[incr-800]]
#     elif incr >= 900 and incr <1000:
#         data0=[data[incr-900]]
#     elif incr >= 1000 and incr <1100:
#         data0=[data[incr-1000]]
#     elif incr >= 1100 and incr <1200:
#         data0=[data[incr-1100]]
#     elif incr >= 1200 and incr <1300:
#         data0=[data[incr-1200]]
#     elif incr >= 1300 and incr <1400:
#         data0=[data[incr-1300]]
#     elif incr >= 1400 and incr <1500:
#         data0=[data[incr-1400]]
#     elif incr >= 1500 and incr <1600:
#         data0=[data[incr-1500]]
#     elif incr >= 1600 and incr <1700:
#         data0=[data[incr-1600]]
#     elif incr >= 1700 and incr <1800:
#         data0=[data[incr-1700]]
#     elif incr >= 1800 and incr <1900:
#         data0=[data[incr-1800]]
#     elif incr >= 1900 and incr <2000:
#         data0=[data[incr-1900]]
#     else:
#         data0=[data[incr]]
# =============================================================================
    #data0=1
    #data1 = np.array([1])
    data1 = np.random.randint(0,2,1)
    print('This is data1 {}'.format(data1))
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


import numpy as np
import matplotlib.pyplot as plt	
episodes=400
solved=0
steps_ep=[]
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]   
q=(6,len(actions_list))
Q1=np.zeros(q)
total_episodes=[]
for episode in range(episodes):
    gamma= 1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset(episode)
    #print(state_n)
    state_n=state_n[0]
    done=False
    reward_episode=[]
    steps=0
    counterr=0
    #for t in range(0,15):
    while done!=True:
        steps+=1
        random_values=Q1[int(state_n[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        print(actiona,np.argmax(state_n[0]))
        q_val=(int(np.argmax(state_n)),actiona)
        print('Print q_val {}'.format(q_val))
        action=np.array(actions_list[actiona])#(actiona,actionb)
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        value=re + gamma * max(Q1[int(stat[0])])
        Q1[(q_val)]=value
        done=do
        print('This is the reward {}'.format(re))
        reward_episode.append(re)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1 
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))

import numpy as np
import matplotlib.pyplot as plt	
episodes=400
solved=0
steps_ep=[]
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]   
q=(6,len(actions_list))
Q2=np.zeros(q)
total_episodes=[]
for episode in range(episodes):
    #np.random.seed(0)
    gamma= 1
    state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset(episode)
    #print(state_n)
    state_n=state_n[0]
    done=False
    reward_episode=[]
    steps=0
    counterr=0
    #for t in range(0,15):
    while done!=True:
        steps+=1
        random_values=Q2[int(state_n[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
        actiona=np.argmax(random_values)
        print(actiona,np.argmax(state_n[0]))
        q_val=(int(np.argmax(state_n)),actiona)
        print('Print q_val {}'.format(q_val))
        action=np.array(actions_list[actiona])#(actiona,actionb)
        stat,re,do,action_h,bob_key=step(action,act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        done=do
        value=re + gamma * max(Q2[int(stat[0])])
        Q2[(q_val)]=value
        print('This is the reward {}'.format(re))
        reward_episode.append(re)
        state_n=stat
    if reward_episode[-1]==1:
        solved+=1 
        steps_ep.append(len(reward_episode))
    total_episodes.append(reward_episode[-1])
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))

total_re=[]
solved=0
episodes = 100
for _ in range(episodes):
    state_n1,actions1,data1,al_coun1,al_data1,bob_count1,bob_mail1,bob_mailbox1,bob_k1,done1,act_hist1,cum_re1,state_space1,action_space1,max_moves1,al_obs1,bob_data1,=reset(_)
    state_n2,actions2,data2,al_coun2,al_data2,bob_count2,bob_mail2,bob_mailbox2,bob_k2,done2,act_hist2,cum_re2,state_space2,action_space2,max_moves2,al_obs2,bob_data2,=reset(_)
#    epochs, penalties, reward = 0, 0, 0
    state_n1=state_n1[0][0]
    state_n2=state_n2[0][0]
    done1 = False
    done2 = False
    while done1!=True or done2!=True:
        action1 = np.argmax(Q1[int(state_n1)])
        action2 = np.argmax(Q2[int(state_n2)])
        #print(action)
        action01=np.array(actions_list[action1])
        action02=np.array(actions_list[action2])
        stat1,re1,do1,action_h1,bob_key1=step(action01,act_hist1,max_moves1,al_coun1,data1,al_data1,bob_count1,al_obs1,bob_k1,bob_data1,cum_re1,bob_mailbox1,bob_mail1,done1, verbose=0,)
        stat2,re2,do2,action_h2,bob_key2=step(action02,act_hist2,max_moves2,al_coun2,data2,al_data2,bob_count2,al_obs2,bob_k2,bob_data2,cum_re2,bob_mailbox2,bob_mail2,done2, verbose=0,)
        done1=do1
        done2=do2
        reward1=re1
        reward2=re2
        print(stat1[0])
        print(stat2[0])
        state_n1=stat1[0]
        state_n2=stat2[0]
        #print(re)
        if do1 ==True or do2 == True:
            if reward1==1 or reward2==1:
                total_re.append(1)
            if re==1:
                solved+=1
print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
