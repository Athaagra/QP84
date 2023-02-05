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
import random
random.seed(0)
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
LogicalStates=np.array([[1,0],[0,1]])
LogicalStates2bit=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
LogicalStates3bit=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
LogicalStates4bit=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
def qtableu():
    import pandas as pd
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    statesColumns=[[-1.,-1.,-1.,-1.],
    [0.,1.,2.,0.],
    [0.,1.,2.,1.],
    [1.,0.,2.,1.],
    [2.,0.,2.,1.],
    [2.,0.,1.,1.],
    [1.,0.,2.,2.],
    [0.,2.,1.,2.],
    [2.,2.,1.,0.],
    [2.,2.,2.,-1.],
    [2.,0.,1.,2.],
    [1.,2.,0.,2.],
    [1.,2.,0.,0.],
    [1.,1.,0.,2.],
    [2.,1.,0.,0.],
    [2.,2.,1.,-1.],
    [2.,1.,0.,1.],
    [2.,0.,2.,-1.],
    [2.,0.,0.,1.],
    [2.,1.,0.,2.],
    [1.,0.,1.,2.],
    [0.,2.,1.,0.],
    [0.,2.,2.,1.],
    [0.,1.,0.,2.],
    [2.,1.,0.,-1.],
    [0.,1.,2.,2.],
    [0.,2.,0.,1.],               
    [1.,2.,0.,1.],
    [0.,0.,1.,2.],
    [2.,1.,2.,0.],
    [0.,1.,1.,-1.],
    [0.,0.,2.,-1.],
    [1.,2.,0.,-1.],
    [1.,0.,1.,-1.],
    [2.,2.,0.,-1.],
    [0.,1.,1.,2.],
    [1.,1.,2.,-1.],
    [1.,0.,2.,-1.],
    [2.,1.,-1.,-1.],
    [2.,1.,1.,-1.],
    [2.,0.,-1.,-1.],
    [0.,2.,2.,-1.],
    [1.,2.,1.,-1.],
    [2.,0.,0.,-1.],
    [1.,0.,0.,-1.],
    [1.,0.,2.,0.],
    [1.,2.,1.,0.],
    [2.,0.,1.,-1.],
    [2.,1.,1.,0.],
    [0.,0.,1.,-1.],
    [2.,2.,0.,1.],
    [0.,2.,1.,1.],
    [2.,1.,2.,-1.],
    [1.,2.,2.,-1.],
    [0.,0.,2.,1.],
    [2.,0.,1.,0.],
    [0.,1.,2.,-1.],
    [1.,2.,2.,0.],
    [0.,2.,0.,-1.],
    [0.,1.,0.,-1.],
    [0.,2.,1.,-1.],
    [1.,0.,-1.,-1.],
    [1.,2.,-1.,-1.],
    [0.,1.,-1.,-1.],
    [1.,1.,0.,-1.],
    [1.,0.,0.,2.],              
    [0.,2.,-1.,-1.],
    [-1.,-1.,-1.,0.], 
    [-1.,-1.,0.,-1.], 
    [-1.,-1.,0.,0.], 
    [-1.,0.,-1.,-1.], 
    [-1.,0.,-1.,0.], 
    [-1.,0.,0.,-1.], 
    [-1.,0.,0.,0.], 
    [0.,-1.,-1.,-1.], 
    [0.,-1.,-1.,0.], 
    [0.,-1.,0.,-1.], 
    [0.,-1.,0.,0.], 
    [0.,0.,-1.,-1.], 
    [0.,0.,-1.,0.], 
    [0.,0.,0.,-1.], 
    [0.,0.,0.,0.], 
    [-1.,-1.,-1.,1.], 
    [-1.,-1.,1.,-1.], 
    [-1.,-1.,1.,1.], 
    [-1.,1.,-1.,-1.], 
    [-1.,1.,-1.,1.], 
    [-1.,1.,1.,-1.], 
    [-1.,1.,1.,1.], 
    [1.,-1.,-1.,-1.], 
    [1.,-1.,-1.,1.], 
    [1.,-1.,1.,-1.], 
    [1.,-1.,1.,1.], 
    [1.,1.,-1.,-1.], 
    [1.,1.,-1.,1.], 
    [1.,1.,1.,-1.], 
    [-1.,-1.,-1.,2.], 
    [-1.,-1.,2.,-1.], 
    [-1.,-1.,2.,2.], 
    [-1.,2.,-1.,-1.], 
    [-1.,2.,-1.,1.], 
    [-1.,2.,2.,-1.], 
    [-1.,2.,2.,2.], 
    [2.,-1.,-1.,-1.], 
    [2.,-1.,-1.,2.], 
    [2.,-1.,2.,-1.], 
    [2.,-1.,2.,2.], 
    [2.,2.,-1.,-1.], 
    [2.,2.,-1.,2.], 
    [0.,0.,0.,2.], 
    [0.,0.,2.,0.], 
    [0.,0.,2.,2.], 
    [0.,2.,0.,0.], 
    [0.,2.,0.,2.], 
    [0.,2.,2.,0.], 
    [0.,2.,2.,2.], 
    [2.,0.,0.,0.], 
    [2.,0.,0.,2.], 
    [2.,0.,2.,0.], 
    [2.,0.,2.,2.], 
    [2.,2.,0.,0.], 
    [2.,2.,0.,2.], 
    [2.,2.,2.,0.], 
    [1.,1.,1.,2.], 
    [1.,1.,2.,1.], 
    [1.,1.,2.,2.], 
    [1.,2.,1.,1.], 
    [1.,2.,1.,2.], 
    [1.,2.,2.,1.], 
    [1.,2.,2.,2.], 
    [2.,1.,1.,1.], 
    [2.,1.,1.,2.], 
    [2.,1.,2.,1.], 
    [2.,1.,2.,2.], 
    [2.,2.,1.,1.], 
    [2.,2.,1.,2.], 
    [2.,2.,2.,1.], 
    [2.,2.,2.,2.], 
    [0.,0.,0.,1.], 
    [0.,0.,1.,0.], 
    [0.,0.,1.,1.], 
    [0.,1.,0.,0.], 
    [0.,1.,0.,1.], 
    [0.,1.,1.,0.], 
    [0.,1.,1.,1.], 
    [1.,0.,0.,0.], 
    [1.,0.,0.,1.], 
    [1.,0.,1.,0.], 
    [1.,0.,1.,1.], 
    [1.,1.,0.,0.], 
    [1.,1.,0.,1.], 
    [1.,1.,1.,0.],
    [1.,1.,2.,0.], 
    [1.,1.,1.,1.]]
    StateColumns=[]
    for i in range(len(statesColumns)):
        bob_keys=''.join(str(int(x)) for x in statesColumns[i][:len(statesColumns[i])])
        StateColumns.append(bob_keys)
    statesColumns=StateColumns
    q=(len(statesColumns),len(actions_list))
    Q=np.zeros(q)
    Qtable=pd.DataFrame(Q.T,columns=statesColumns)
    return Qtable
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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 13))
    plt.bar(1,pvalue)
    plt.xlabel(f'Mannwhitney Test')
    plt.ylabel('Probability')
    plt.title(str(resultss))#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return resultss
def Qlearning(Qtable,inp,ma=False,qp=False,onebit=False,twobit=False,threebit=False,fourbit=False,gamma_v=0.001,learning_ra=1e-3):
    import numpy as np
    episodes=500
    solved=0
    #actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    steps_ep=[]
    total_episodes=[]
    total_fidelity=[]
    cum=[]
    cumre=0
    testing=[]
    staates=[]
    for episode in range(episodes):
        import matplotlib.pyplot as plt
        gamma= gamma_v
        learning_rate=learning_ra 
        env=Qprotocol(4,inp,MultiAgent=ma,Qb=qp)
        state_n,inpu=env.reset(4)
        #state_n=str((state_n[0][0],state_n[0][1],state_n[0][2],state_n[0][3]))
        done=False
        reward_episode=[]
        steps=0
        reward=0
        while done!=True:
            q_val=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            print(q_val)
            steps+=1
            ravar=np.random.randint(11, size=(1,len(actions_list)))/1000
            random_values=Qtable.loc[:,q_val] + ravar[0]#Qtable[str(state_n[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            print('This is the random value {}'.format(random_values))
            actiona=np.argmax(random_values)
            action=np.array(actions_list[actiona])
            stat,reward,done,action_h,bob_key=env.step(action)
            q_val_n=''.join(str(int(x)) for x in stat[0][:len(stat[0])])#str((int(stat[0][0]),int(stat[0][1]),int(stat[0][2]),int(stat[0][3])))
            print('the q_val {},the q_val_n {}, this is the action {}'.format(q_val,q_val_n,actiona))
            Qtable[q_val][actiona]=(1-learning_rate)*Qtable[q_val][actiona]+learning_rate * (reward + gamma * max(Qtable.loc[:,q_val_n]) - Qtable[q_val][actiona])
            testing.append([action,reward])
            #print('This is the tabular {}'.format(Q))
            reward_episode.append(reward)
            state_n=stat
        if done==True:
            cumre+=reward
            cum.append(cumre)
            if len(bob_key)==len(inpu):
                if onebit==True:
                    tp=LogicalStates[:,inpu].T*LogicalStates[bob_key,:]
                    tp=tp[0]
                    inpus=str(inpu)
                    Fidelity=abs(sum(tp))**2
                if twobit==True:
                    inpus=''.join(str(x) for x in inpu)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inpu)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                if threebit==True:
                    inpus=''.join(str(x) for x in inpu)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inpu)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                if fourbit==True:
                    inpus=''.join(str(x) for x in inpu)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inpu)])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
            else:
                Fidelity=0
            if reward_episode[-1]==1:
                solved+=1
                steps_ep.append(len(reward_episode))
            else:
                solved+=0
                steps_ep.append(len(reward_episode))
        total_episodes.append(reward_episode[-1])
        total_fidelity.append(Fidelity)
    print('The simulation has been solved the environment '+inpus+' Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+inpus+' Q learning:{}'.format(solved/episodes))
    plt.show()
    
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Fidelity')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+inpus+' Q learning fidelity:{}'.format(sum(total_fidelity)))
    plt.show() 
    #print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
    #print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(steps_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps per episode '+inpus+' that solved:{}'.format(np.round(np.mean(steps_ep))))
    plt.show()
    
    plt.figure(figsize=(13, 13))
    plt.plot(cum)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Cumulative Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+inpus+' Q learning Cumulative:{}'.format(cum[-1]))
    plt.show()
    error=env.error_counter
    results=mannwhitney(total_episodes,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(max(cum)),'Steps:'+str(np.mean(steps_ep)),'Fidelity:'+str(sum(total_fidelity))])
    return Qtable,results

def OenvOagentsimulation(Qt,inp,qp):
    total_re=[]
    steps_per_ep=[]
    solved=0
    cum=[]
    total_fid=[]
    episodes = 100
    for ep in range(episodes):
        env=Qprotocol(4,inp,Qb=qp)
        state_n,inpu=env.reset(4)
        print('This is the inpu {}'.format(inpu))
        done = False
        steps=0
        while done!=True:
            state_n=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            action1 = np.argmax(Qt[state_n])
            action1=np.array(actions_list[action1])
            stat,reward,done,action_h,bob_key=env.step(action1)
            steps+=1
            state_n=stat
            if done==True:
                bke=bob_key
            if done==True:
                if len(inpu)==1 and len(bke)==len(inpu):
                    tp=LogicalStates[:,inpu].T*LogicalStates[bke,:]
                    tp=tp[0]
                    inpus=str(inpu)
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if len(inpu)==2 and len(bke)==len(inpu):
                    inpus=''.join(str(x) for x in inpu)
                    bk=''.join(str(x) for x in bke[:len(inpu)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bk,:])
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if len(inpu)==3 and len(bke)==len(inpu):
                    inpus=''.join(str(x) for x in inpu)
                    bk=''.join(str(x) for x in bke[:len(inpu)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bk,:])
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if len(inpu)==4 and len(bke)==len(inpu):
                    inpus=''.join(str(x) for x in inpu)
                    bk=''.join(str(x) for x in bke[:len(inpu)])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bk,:])
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if reward==1:
                    total_re.append(1)                
                    steps_per_ep.append(steps)
                    solved+=1
                    cum.append(solved)
                    break
                else:
                    total_re.append(0)
                    steps_per_ep.append(steps)
                    solved+=0
                    cum.append(solved)
                    break
    import matplotlib.pyplot as plt
    print('The simulation has been solved the '+str(inp)+' environment Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode '+str(inp)+' that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_re)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total rewards environment '+str(inp)+' per episode :{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fid)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity environment '+str(inp)+' per episode :{}'.format(sum(total_fid)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_per_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps per episode environment '+str(inp)+' that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.show()
    error=env.error_counter
    results=mannwhitney(total_re,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(cum[-1]),'Steps:'+str(np.mean(steps_per_ep)),'Fidelity:'+str(sum(total_fid))])
    return results




def onebitsimulation(Qt,Qt1,qp):
    total_re=[]
    steps_per_ep=[]
    solved=0
    total_fid=[]
    cum=[]
    episodes = 100
    for ep in range(episodes):
        inp=np.random.randint(0,2,1)
        env=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env1=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        state_n,inpu=env.reset(4)
        state_n1,inpu=env1.reset(4)
        done = False
        done1=False
        steps=0
        print(inpu)
        while done!=True or done1!=True:
            state_n=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            state_n1=''.join(str(int(x)) for x in state_n1[0][:len(state_n1[0])])
            action1 = np.argmax(Qt[state_n])
            action2 = np.argmax(Qt1[state_n1])
            action1=np.array(actions_list[action1])
            action2=np.array(actions_list[action2])
            stat,reward,done,action_h,bob_key=env.step(action1)
            stat1,reward1,done1,action_h1,bob_key1=env1.step(action2)
            steps+=1
            state_n=stat
            state_n1=stat1
            if done==True:
                bk=bob_key
            if done1==True:
                bk=bob_key1
            if done==True or done1==True:
                tp=LogicalStates[:,inpu].T*LogicalStates[bk,:]
                tp=tp[0]
                inpus=str(inpu)
                Fidelity=abs(sum(tp))**2
                total_fid.append(Fidelity)
                if reward==1 or reward1==1:
                    total_re.append(1)                
                    steps_per_ep.append(steps)
                    solved+=1
                    cum.append(solved)
                    break
                else:
                    total_re.append(0)
                    steps_per_ep.append(steps)
                    solved+=0
                    cum.append(solved)
                    break
    import matplotlib.pyplot as plt
    print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_re)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total rewards '+inpus+' per episode :{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fid)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity '+inpus+' per episode :{}'.format(sum(total_fid)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_per_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps '+inpus+' per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    results=mannwhitney(total_re,error)
    results1=mannwhitney(total_re,error1)
    results.append([results1,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum[-1]),'Steps:'+str(np.mean(steps_per_ep)),'Fidelity:'+str(sum(total_fid))])
    return results

def twobitsimulation(Qt,Qt1,Qt2,Qt3,qp):
    total_re=[]
    steps_per_ep=[]
    solved=0
    total_fid=[]
    cum=[]
    episodes = 100
    for ep in range(episodes):
        inp=np.random.randint(0,2,2)
        env=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env1=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env2=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env3=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        state_n,inpu=env.reset(4)
        state_n1,inpu1=env1.reset(4)
        state_n2,inpu2=env2.reset(4)
        state_n3,inpu3=env3.reset(4)
        print('This is the initial message {},{},{},{}'.format(inpu,inpu1,inpu2,inpu3))
        done = False
        done1=False
        done2=False
        done3=False
        steps=0
        while done!=True or done1!=True or done2!=True or done3!=True:
            state_n=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            state_n1=''.join(str(int(x)) for x in state_n1[0][:len(state_n1[0])])
            state_n2=''.join(str(int(x)) for x in state_n2[0][:len(state_n2[0])])
            state_n3=''.join(str(int(x)) for x in state_n3[0][:len(state_n3[0])])
            action1 = np.argmax(Qt[state_n])
            action2 = np.argmax(Qt1[state_n1])
            action3 = np.argmax(Qt2[state_n2])
            action4 = np.argmax(Qt3[state_n3])
            action1=np.array(actions_list[action1])
            action2=np.array(actions_list[action2])
            action3=np.array(actions_list[action3])
            action4=np.array(actions_list[action4])
            stat,reward,done,action_h,bob_key=env.step(action1)
            stat1,reward1,done1,action_h1,bob_key1=env1.step(action2)
            stat2,reward2,done2,action_h2,bob_key2=env2.step(action3)
            stat3,reward3,done3,action_h3,bob_key3=env3.step(action4)
            steps+=1
            state_n=stat
            state_n1=stat1
            state_n2=stat2
            state_n3=stat3
            if done==True:
                bk=bob_key
            if done1==True:
                bk=bob_key1
            if done2==True:
                bk=bob_key2
            if done3==True:
                bk=bob_key3
            if done==True or done1==True or done2==True or done3==True:# or do7==True or do8==True or do9 ==True or do10 == True or do11==True or do12==True or do13==True or do14==True:# or do15==True or do16==True:
                #tp=LogicalStates[:,inpu].T*LogicalStates[bk,:]
                #tp=tp[0]
                inpus=''.join(str(x) for x in inpu)
                bk=''.join(str(x) for x in bk[:len(inpu)])
                tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bk,:])
                Fidelity=abs(sum(tp))**2
                total_fid.append(Fidelity)
                if reward==1 or reward1==1 or reward2==1 or reward3==1:
                    total_re.append(1)                
                    steps_per_ep.append(steps)
                    solved+=1
                    cum.append(solved)
                    break
                else:
                    total_re.append(0)
                    steps_per_ep.append(steps)
                    solved+=0
                    cum.append(solved)
                    break
    import matplotlib.pyplot as plt
    print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_re)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total rewards per episode :{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fid)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity per episode :{}'.format(sum(total_fid)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_per_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    results=mannwhitney(total_re,error)
    results1=mannwhitney(total_re,error1)
    results2=mannwhitney(total_re,error2)
    results3=mannwhitney(total_re,error3)
    results.append([results1,results2,results3,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum[-1]),'Steps:'+str(np.mean(steps_per_ep)),'Fidelity:'+str(sum(total_fid))])
    return results

def threebitsimulation(Qt,Qt1,Qt2,Qt3,Qt4,Qt5,Qt6,Qt7,qp):
    total_re=[]
    cum=[]
    steps_per_ep=[]
    solved=0
    total_fid=[]
    episodes = 100
    for ep in range(episodes):
        inp=np.random.randint(0,2,3)
        #print(inpu)
        env=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env1=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env2=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env3=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env4=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env5=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env6=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env7=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        state_n,inpu=env.reset(4)
        state_n1,inpu1=env1.reset(4)
        state_n2,inpu2=env2.reset(4)
        state_n3,inpu3=env3.reset(4)
        state_n4,inpu4=env4.reset(4)
        state_n5,inpu5=env5.reset(4)
        state_n6,inpu6=env6.reset(4)
        state_n7,inpu7=env7.reset(4)
        print(inpu,inpu1,inpu2,inpu3,inpu4,inpu5,inpu6,inpu7)
        done = False
        done1=False
        done2=False
        done3=False
        done4=False
        done5=False
        done6=False
        done7=False
        steps=0
        while done!=True or done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True:# or done8!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
            state_n=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            state_n1=''.join(str(int(x)) for x in state_n1[0][:len(state_n1[0])])
            state_n2=''.join(str(int(x)) for x in state_n2[0][:len(state_n2[0])])
            state_n3=''.join(str(int(x)) for x in state_n3[0][:len(state_n3[0])])
            state_n4=''.join(str(int(x)) for x in state_n4[0][:len(state_n4[0])])
            state_n5=''.join(str(int(x)) for x in state_n5[0][:len(state_n5[0])])
            state_n6=''.join(str(int(x)) for x in state_n6[0][:len(state_n6[0])])
            state_n7=''.join(str(int(x)) for x in state_n7[0][:len(state_n7[0])])
            action1 = np.argmax(Qt[state_n])
            action2 = np.argmax(Qt1[state_n1])
            action3 = np.argmax(Qt2[state_n2])
            action4 = np.argmax(Qt3[state_n3])
            action5 = np.argmax(Qt4[state_n4])
            action6 = np.argmax(Qt5[state_n5])
            action7 = np.argmax(Qt6[state_n6])
            action8 = np.argmax(Qt7[state_n7])
            action1=np.array(actions_list[action1])
            action2=np.array(actions_list[action2])
            action3=np.array(actions_list[action3])
            action4=np.array(actions_list[action4])
            action5=np.array(actions_list[action5])
            action6=np.array(actions_list[action6])
            action7=np.array(actions_list[action7])
            action8=np.array(actions_list[action8])
            stat,reward,done,action_h,bob_key=env.step(action1)
            stat1,reward1,done1,action_h1,bob_key1=env1.step(action2)
            stat2,reward2,done2,action_h2,bob_key2=env2.step(action3)
            stat3,reward3,done3,action_h3,bob_key3=env3.step(action4)
            stat4,reward4,done4,action_h4,bob_key4=env4.step(action5)
            stat5,reward5,done5,action_h5,bob_key5=env5.step(action6)
            stat6,reward6,done6,action_h6,bob_key6=env6.step(action7)
            stat7,reward7,done7,action_h7,bob_key7=env7.step(action8)
            steps+=1
            state_n=stat
            state_n1=stat1
            state_n2=stat2
            state_n3=stat3
            state_n4=stat4
            state_n5=stat5
            state_n6=stat6
            state_n7=stat7
            if done==True:
                bk=bob_key
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
            if done==True or done1==True or done2==True or done3==True or done4==True or done5==True or done6==True or done7==True:# or do7==True or do8==True or do9 ==True or do10 == True or do11==True or do12==True or do13==True or do14==True:# or do15==True or do16==True:
                #tp=LogicalStates[:,inpu].T*LogicalStates[bk,:]
                #tp=tp[0]
                if len(bk)==len(inpu):
                    inpus=''.join(str(x) for x in inpu)
                    bk=''.join(str(x) for x in bk[:len(inpu)])
                    #tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bk,:])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bk,:])
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if reward==1 or reward1==1 or reward2==1 or reward3==1 or  reward5==1 or reward6==1 or  reward7==1:
                    total_re.append(1)                
                    steps_per_ep.append(steps)
                    solved+=1
                    cum.append(solved)
                    break
                else:
                    total_re.append(0)
                    steps_per_ep.append(steps)
                    solved+=0
                    cum.append(solved)
                    break
                #if re==1:
    import matplotlib.pyplot as plt
    print('The simulation has been solved the '+str(len(inp))+' environment Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode '+str(len(inp))+' that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_re)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total rewards per episode '+str(len(inp))+' :{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fid)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity per episode '+str(len(inp))+':{}'.format(sum(total_fid)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_per_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps per episode '+str(len(inp))+' that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    error4=env4.error_counter
    error5=env5.error_counter
    error6=env6.error_counter
    error7=env7.error_counter
    results=mannwhitney(total_re,error)
    results1=mannwhitney(total_re,error1)
    results2=mannwhitney(total_re,error2)
    results3=mannwhitney(total_re,error3)
    results4=mannwhitney(total_re,error4)
    results5=mannwhitney(total_re,error5)
    results6=mannwhitney(total_re,error6)
    results7=mannwhitney(total_re,error7)
    results.append([results1,results2,results3,results4,results5,results6,results7,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum[-1]),'Steps:'+str(np.mean(steps_per_ep)),'Fidelity:'+str(sum(total_fid))])
    return results


def fourbitsimulation(Qt,Qt1,Qt2,Qt3,Qt4,Qt5,Qt6,Qt7,Qt8,Qt9,Qt10,Qt11,Qt12,Qt13,Qt14,Qt15,qp):
    total_re=[]
    steps_per_ep=[]
    solved=0
    cum=[]
    total_fid=[]
    episodes = 100
    for ep in range(episodes):
        inp=np.random.randint(0,2,4)
        env=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env1=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env2=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env3=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env4=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env5=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env6=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env7=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env8=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env9=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env10=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env11=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env12=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env13=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env14=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        env15=Qprotocol(4,inp,MultiAgent=True,Qb=qp)
        state_n,inpu=env.reset(4)
        state_n1,inpu1=env1.reset(4)
        state_n2,inpu2=env2.reset(4)
        state_n3,inpu3=env3.reset(4)
        state_n4,inpu4=env4.reset(4)
        state_n5,inpu5=env5.reset(4)
        state_n6,inpu6=env6.reset(4)
        state_n7,inpu7=env7.reset(4)
        state_n8,inpu8=env8.reset(4)
        state_n9,inpu9=env9.reset(4)
        state_n10,inpu10=env10.reset(4)
        state_n11,inpu11=env11.reset(4)
        state_n12,inpu12=env12.reset(4)
        state_n13,inpu13=env13.reset(4)
        state_n14,inpu14=env14.reset(4)
        state_n15,inpu15=env15.reset(4)
        done=False
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
        steps=0
        while done!=True or done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True:# or done15!=True or done16!=True:
            state_n=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
            state_n1=''.join(str(int(x)) for x in state_n1[0][:len(state_n1[0])])
            state_n2=''.join(str(int(x)) for x in state_n2[0][:len(state_n2[0])])
            state_n3=''.join(str(int(x)) for x in state_n3[0][:len(state_n3[0])])
            state_n4=''.join(str(int(x)) for x in state_n4[0][:len(state_n4[0])])
            state_n5=''.join(str(int(x)) for x in state_n5[0][:len(state_n5[0])])
            state_n6=''.join(str(int(x)) for x in state_n6[0][:len(state_n6[0])])
            state_n7=''.join(str(int(x)) for x in state_n7[0][:len(state_n7[0])])
            state_n8=''.join(str(int(x)) for x in state_n8[0][:len(state_n8[0])])
            state_n9=''.join(str(int(x)) for x in state_n9[0][:len(state_n9[0])])
            state_n10=''.join(str(int(x)) for x in state_n10[0][:len(state_n10[0])])
            state_n11=''.join(str(int(x)) for x in state_n11[0][:len(state_n11[0])])
            state_n12=''.join(str(int(x)) for x in state_n12[0][:len(state_n12[0])])
            state_n13=''.join(str(int(x)) for x in state_n13[0][:len(state_n13[0])])
            state_n14=''.join(str(int(x)) for x in state_n14[0][:len(state_n14[0])])
            state_n15=''.join(str(int(x)) for x in state_n15[0][:len(state_n15[0])])
            action1 = np.argmax(Qt[state_n])
            action2 = np.argmax(Qt1[state_n1])
            action3 = np.argmax(Qt2[state_n2])
            action4 = np.argmax(Qt3[state_n3])
            action5 = np.argmax(Qt4[state_n4])
            action6 = np.argmax(Qt5[state_n5])
            action7 = np.argmax(Qt6[state_n6])
            action8 = np.argmax(Qt7[state_n7])
            action9 = np.argmax(Qt8[state_n8])
            action10 = np.argmax(Qt9[state_n9])
            action11 = np.argmax(Qt10[state_n10])
            action12 = np.argmax(Qt11[state_n11])
            action13 = np.argmax(Qt12[state_n12])
            action14 = np.argmax(Qt13[state_n13])
            action15 = np.argmax(Qt14[state_n14])
            action16 = np.argmax(Qt15[state_n15])
            action1=np.array(actions_list[action1])
            action2=np.array(actions_list[action2])
            action3=np.array(actions_list[action3])
            action4=np.array(actions_list[action4])
            action5=np.array(actions_list[action5])
            action6=np.array(actions_list[action6])
            action7=np.array(actions_list[action7])
            action8=np.array(actions_list[action8])
            action9=np.array(actions_list[action9])
            action10=np.array(actions_list[action10])
            action11=np.array(actions_list[action11])
            action12=np.array(actions_list[action12])
            action13=np.array(actions_list[action13])
            action14=np.array(actions_list[action14])
            action15=np.array(actions_list[action15])
            action16=np.array(actions_list[action16])
            stat,reward,done,action_h,bob_key=env.step(action1)
            stat1,reward1,done1,action_h1,bob_key1=env1.step(action2)
            stat2,reward2,done2,action_h2,bob_key2=env2.step(action3)
            stat3,reward3,done3,action_h3,bob_key3=env3.step(action4)
            stat4,reward4,done4,action_h4,bob_key4=env4.step(action5)
            stat5,reward5,done5,action_h5,bob_key5=env5.step(action6)
            stat6,reward6,done6,action_h6,bob_key6=env6.step(action7)
            stat7,reward7,done7,action_h7,bob_key7=env7.step(action8)
            stat8,reward8,done8,action_h8,bob_key8=env8.step(action9)
            stat9,reward9,done9,action_h9,bob_key9=env9.step(action10)
            stat10,reward10,done10,action_h10,bob_key10=env10.step(action11)
            stat11,reward11,done11,action_h11,bob_key11=env11.step(action12)
            stat12,reward12,done12,action_h12,bob_key12=env12.step(action13)
            stat13,reward13,done13,action_h13,bob_key13=env13.step(action14)
            stat14,reward14,done14,action_h14,bob_key14=env14.step(action15)
            stat15,reward15,done15,action_h15,bob_key15=env15.step(action16)
            steps+=1
            state_n=stat
            state_n1=stat1
            state_n2=stat2
            state_n3=stat3
            state_n4=stat4
            state_n5=stat5
            state_n6=stat6
            state_n7=stat7
            state_n8=stat8
            state_n9=stat9
            state_n10=stat10
            state_n11=stat11
            state_n12=stat12
            state_n13=stat13
            state_n14=stat14
            state_n15=stat15
            if done==True:
                bk=bob_key
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
            if done==True or done1==True or done2==True or done3==True or done4==True or done5==True or done6==True or done7==True or done8==True or done9 ==True or done10 == True or done11==True or done12==True or done13==True or done14==True or done15==True:
                inpus=''.join(str(x) for x in inpu)
                print('This is input {}'.format(inpus))
                bok=''.join(str(x) for x in bk[:len(inpu)])
                print('This is the bk {}'.format(bok))
                if len(bk)==len(inpu):
                    #tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bk,:])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bok,:])
                    Fidelity=abs(sum(tp))**2
                    total_fid.append(Fidelity)
                else:
                    total_fid.append(0)
                if reward==1 or reward1==1 or reward2==1 or reward3==1 or  reward5==1 or reward6==1 or  reward7==1 or reward8==1 or reward9==1 or reward10==1 or reward11==1 or reward12==1 or  reward13==1 or reward14==1 or reward15==1:
                    total_re.append(1)                
                    steps_per_ep.append(steps)
                    solved+=1
                    cum.append(solved)
                    break
                else:
                    total_re.append(0)
                    steps_per_ep.append(steps)
                    solved+=0
                    cum.append(solved)
                    break
    import matplotlib.pyplot as plt
    print('The simulation has been solved the environment Q learning:{}'.format(solved/episodes))
    print('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
    plt.figure(figsize=(13, 13))
    plt.plot(total_re)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total rewards per episode :{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fid)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('Total fidelity per episode :{}'.format(sum(total_fid)))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_per_ep)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Number of Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The number of steps per episode that solved:{}'.format(np.round(np.mean(steps_per_ep))))
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
    results=mannwhitney(total_re,error)
    results1=mannwhitney(total_re,error1)
    results2=mannwhitney(total_re,error2)
    results3=mannwhitney(total_re,error3)
    results4=mannwhitney(total_re,error4)
    results5=mannwhitney(total_re,error5)
    results6=mannwhitney(total_re,error6)
    results7=mannwhitney(total_re,error7)
    results8=mannwhitney(total_re,error8)
    results9=mannwhitney(total_re,error9)
    results10=mannwhitney(total_re,error10)
    results11=mannwhitney(total_re,error11)
    results12=mannwhitney(total_re,error12)
    results13=mannwhitney(total_re,error13)
    results14=mannwhitney(total_re,error14)
    results15=mannwhitney(total_re,error15)
    results.append([results1,results2,results3,results4,results5,results6,results7,results8,results9,results10,results11,results12,results13,results14,results15,'Reward:'+str(solved/episodes),'Cumulative:'+str(cum[-1]),'Steps:'+str(np.mean(steps_per_ep)),'Fidelity:'+str(sum(total_fid))])
    return results


Qtable=qtableu()
Q,r=Qlearning(Qtable,1,onebit=True)
print(r,file=open('randomOneBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,1,qp=False)
print(r,file=open('randomOneBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,2,twobit=True,gamma_v=0.0001,learning_ra=1e-3)
print(r,file=open('randomTwoBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,2,qp=False)
print(r,file=open('randomTwoBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,3,threebit=True)
print(r,file=open('randomThreeBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,3,qp=False)
print(r,file=open('randomThreeBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,4,fourbit=True)
print(r,file=open('randomFourBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,4,False)
print(r,file=open('randomFourBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0],True,onebit=True)
print(r,file=open('randomOne[0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[1],True,onebit=True)
print(r,file=open('randomOne[1]BitQlTraining.txt','w'))
r=onebitsimulation(Q,Q1,False)
print(r,file=open('randomOneMULTIBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0],True,twobit=True)
print(r,file=open('randomTwo[0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,1],True,twobit=True)
print(r,file=open('randomTwo[0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[1,0],True,twobit=True)
print(r,file=open('randomTwo[1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[1,1],True,twobit=True)
print(r,file=open('randomTwo[1,1]BitQlTraining.txt','w'))
r=twobitsimulation(Q,Q1,Q2,Q3,False)
print(r,file=open('randomTwoMULTIBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0,0],True,threebit=True)
print(r,file=open('randomThree[0,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,0,1],True,threebit=True)
print(r,file=open('randomThree[0,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[0,1,0],True,threebit=True)
print(r,file=open('randomThree[0,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[0,1,1],True,threebit=True)
print(r,file=open('randomThree[0,1,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q4,r=Qlearning(Qtable,[1,0,0],True,threebit=True)
print(r,file=open('randomThree[1,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q5,r=Qlearning(Qtable,[1,0,1],True,threebit=True)
print(r,file=open('randomThree[1,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q6,r=Qlearning(Qtable,[1,1,0],True,threebit=True)
print(r,file=open('randomThree[1,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q7,r=Qlearning(Qtable,[1,1,1],True,threebit=True)
print(r,file=open('randomThree[1,1,1]BitQlTraining.txt','w'))
r=threebitsimulation(Q,Q1,Q2,Q3,Q4,Q5,Q6,Q7,False)
print(r,file=open('randomThreeMULTIBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0,0,0],True,fourbit=True)
print(r,file=open('randomFour[0,0,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,0,0,1],True,fourbit=True)
print(r,file=open('randomFour[0,0,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[0,0,1,0],True,fourbit=True)
print(r,file=open('randomFour[0,0,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[0,0,1,1],True,fourbit=True)
print(r,file=open('randomFour[0,0,1,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q4,r=Qlearning(Qtable,[0,1,0,0],True,fourbit=True)
print(r,file=open('randomFour[0,1,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q5,r=Qlearning(Qtable,[0,1,0,1],True,fourbit=True)
print(r,file=open('randomFour[0,1,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q6,r=Qlearning(Qtable,[0,1,1,0],True,fourbit=True)
print(r,file=open('randomFour[0,1,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q7,r=Qlearning(Qtable,[0,1,1,1],True,fourbit=True)
print(r,file=open('randomFour[0,1,1,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q8,r=Qlearning(Qtable,[1,0,0,0],True,fourbit=True)
print(r,file=open('randomFour[1,0,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q9,r=Qlearning(Qtable,[1,0,0,1],True,fourbit=True)
print(r,file=open('randomFour[1,0,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q10,r=Qlearning(Qtable,[1,0,1,0],True,fourbit=True)
print(r,file=open('randomFour[1,0,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q11,r=Qlearning(Qtable,[1,0,1,1],True,fourbit=True)
print(r,file=open('randomFour[1,0,1,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q12,r=Qlearning(Qtable,[1,1,0,0],True,fourbit=True)
print(r,file=open('randomFour[1,1,0,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q13,r=Qlearning(Qtable,[1,1,0,1],True,fourbit=True)
print(r,file=open('randomFour[1,1,0,1]BitQlTraining.txt','w'))
Qtable=qtableu()
Q14,r=Qlearning(Qtable,[1,1,1,0],True,fourbit=True)
print(r,file=open('randomFour[1,1,1,0]BitQlTraining.txt','w'))
Qtable=qtableu()
Q15,r=Qlearning(Qtable,[1,1,1,1],True,fourbit=True)
print(r,file=open('randomFour[1,1,1,1]BitQlTraining.txt','w'))
r=fourbitsimulation(Q,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15,False)
print(r,file=open('randomFourMULTIBitqlTesting.txt','w'))

# =============================================================================
#   Quantum Protocol 
# =============================================================================
Qtable=qtableu()
Q,r=Qlearning(Qtable,1,qp=True,onebit=True)
print(r,file=open('randomOneQBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,1,qp=True)
print(r,file=open('randomOneQBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,2,qp=True,twobit=True,gamma_v=0.0001,learning_ra=1e-3)
print(r,file=open('randomTwoQBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,2,qp=True)
print(r,file=open('randomTwoQBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,3,qp=True,threebit=True)
print(r,file=open('randomThreeBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,3,qp=True)
print(r,file=open('randomThreeQBitQlTesting.txt','w'))
Qtable=qtableu()
Q,r=Qlearning(Qtable,4,qp=True,fourbit=True)
print(r,file=open('randomFourQBitQlTraining.txt','w'))
r=OenvOagentsimulation(Q,4,qp=True)
print(r,file=open('randomFourQBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0],True,qp=True,onebit=True)
print(r,file=open('randomOne[0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[1],True,qp=True,onebit=True)
print(r,file=open('randomOne[1]QBitQlTraining.txt','w'))
r=onebitsimulation(Q,Q1,True)
print(r,file=open('randomOneMULTIQBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0],True,qp=True,twobit=True)
print(r,file=open('randomTwo[0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,1],True,qp=True,twobit=True)
print(r,file=open('randomTwo[0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[1,0],True,qp=True,twobit=True)
print(r,file=open('randomTwo[1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[1,1],True,qp=True,twobit=True)
print(r,file=open('randomTwo[1,1]QBitQlTraining.txt','w'))
r=twobitsimulation(Q,Q1,Q2,Q3,True)
print(r,file=open('randomTwoMULTIQBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0,0],True,qp=True,threebit=True)
print(r,file=open('randomThree[0,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,0,1],True,qp=True,threebit=True)
print(r,file=open('randomThree[0,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[0,1,0],True,qp=True,threebit=True)
print(r,file=open('randomThree[0,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[0,1,1],True,qp=True,threebit=True)
print(r,file=open('randomThree[0,1,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q4,r=Qlearning(Qtable,[1,0,0],True,qp=True,threebit=True)
print(r,file=open('randomThree[1,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q5,r=Qlearning(Qtable,[1,0,1],True,qp=True,threebit=True)
print(r,file=open('randomThree[1,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q6,r=Qlearning(Qtable,[1,1,0],True,qp=True,threebit=True)
print(r,file=open('randomThree[1,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q7,r=Qlearning(Qtable,[1,1,1],True,qp=True,threebit=True)
print(r,file=open('randomThree[1,1,1]QBitQlTraining.txt','w'))
r=threebitsimulation(Q,Q1,Q2,Q3,Q4,Q5,Q6,Q7,True)
print(r,file=open('randomThreeMULTIQBitQlTesting.txt','w'))


Qtable=qtableu()
Q,r=Qlearning(Qtable,[0,0,0,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,0,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q1,r=Qlearning(Qtable,[0,0,0,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,0,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q2,r=Qlearning(Qtable,[0,0,1,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,0,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q3,r=Qlearning(Qtable,[0,0,1,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,0,1,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q4,r=Qlearning(Qtable,[0,1,0,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,1,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q5,r=Qlearning(Qtable,[0,1,0,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,1,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q6,r=Qlearning(Qtable,[0,1,1,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,1,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q7,r=Qlearning(Qtable,[0,1,1,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[0,1,1,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q8,r=Qlearning(Qtable,[1,0,0,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,0,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q9,r=Qlearning(Qtable,[1,0,0,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,0,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q10,r=Qlearning(Qtable,[1,0,1,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,0,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q11,r=Qlearning(Qtable,[1,0,1,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,0,1,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q12,r=Qlearning(Qtable,[1,1,0,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,1,0,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q13,r=Qlearning(Qtable,[1,1,0,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,1,0,1]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q14,r=Qlearning(Qtable,[1,1,1,0],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,1,1,0]QBitQlTraining.txt','w'))
Qtable=qtableu()
Q15,r=Qlearning(Qtable,[1,1,1,1],True,qp=True,fourbit=True)
print(r,file=open('randomFour[1,1,1,1]QBitQlTraining.txt','w'))
r=fourbitsimulation(Q,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15,True)
print(r,file=open('randomFourMULTIQBitqlTesting.txt','w'))
