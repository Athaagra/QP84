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
             self.data1=[self.data1]
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
         if( action_alice == 3 ):
             print('This is the data1 {} and encode'.format(self.data1))
             charsEn="X"
             self.data1=[self.data1]
             if len(self.data1)==1 and type(self.data1[-1])!=str:
                 if self.data1==0:
                     self.data1='+'
                 else:
                     self.data1='-'
             if len(self.data1)==2 and type(self.data1[-1])!=str:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='+'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                         self.data1[typ]='-'
                     
             if len(self.data1)==3 and type(self.data1[-1])!=str:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='+'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                         self.data1[typ]='-'
             if len(self.data1)==4 and type(self.data1[-1])!=str:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='+'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                        self.data1[typ]='-'
         if( action_alice == 4 ):
             print('This is the data1 {} and encode'.format(self.data1))
             charsEn="Z"
             self.data1=[self.data1]
             if len(self.data1)==1 and type(self.data1[-1])!=str and type(self.data1)!=int:
                 if self.data1==0:
                     self.data1='0'
                 else:
                     self.data1='1'
             elif len(self.data1)==2 and type(self.data1[-1])!=str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='0'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                         self.data1[typ]='1'
             elif len(self.data1)==3 and type(self.data1[-1])!=str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='0'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                         self.data1[typ]='1'
             elif len(self.data1)==4 and type(self.data1[-1])!=str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])==str:
                     typ+=1
                     if type(self.data1[typ])!=str and self.data1[typ]==0:
                         self.data1[typ]='0'
                     elif type(self.data1[typ])!=str and self.data1[typ]==1:
                         self.data1[typ]='1'
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
                         print('This is the Bod datalog {}'.format(self.bob_datalog))
         if verbose:
             print("Bob added to his datalog ", self.bob_datalog)
             # Add 0 to key - Bob should decide to take this action based on his datalog
         if( action_bob == 2 ):
             charsEn="X"
             self.data1=[self.data1]
             if len(self.data1)==1 and type(self.data1[-1])==str and type(self.data1)!=int:
                 if self.data1=='-' or self.data1=='1':
                     self.data1=1
                 else:
                     self.data1=0
             elif len(self.data1)==2 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='1':
                         self.data1[typ]=1
                     elif type(self.data1[typ])==str and self.data1[typ]=='+' or self.data1[typ]=='0':
                         self.data1[typ]=0
             elif len(self.data1)==3 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='1':
                         self.data1[typ]=1
                     if type(self.data1[typ])==str and self.data1[typ]=='+' or self.data1[typ]=='0':
                         self.data1[typ]=0
             elif len(self.data1)==4 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='1':
                         self.data1[typ]=1
                     if type(self.data1[typ])==str and self.data1[typ]=='+' or self.data1[typ]=='0':
                         self.data1[typ]=0
         if( action_bob == 3 ):
             charsEn="Z"
             self.data1=[self.data1]
             if len(self.data1)==1 and type(self.data1[-1])==str and type(self.data1)!=int:
                 if self.data1=='1':
                     self.data1=1
                 elif self.data1=='-' or self.data1=='+':
                     self.data1=np.random.randint(0,2,1)[0]
                 else:
                     self.data1=0
             elif len(self.data1)==2 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='1':
                         self.data1[typ]=1
                     elif type(self.data1[typ])==str and self.data1[typ]=='0':
                        self.data1[typ]=0
                     elif type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='+':
                        self.data1[typ]=np.random.randint(0,2,1)[0]
             elif len(self.data1)==3 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='1':
                         self.data1[typ]=1
                     elif type(self.data1[typ])==str and self.data1[typ]=='0':
                        self.data1[typ]=0
                     elif type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='+':
                        self.data1[typ]=np.random.randint(0,2,1)[0]
             elif len(self.data1)==4 and type(self.data1[-1])==str and type(self.data1)!=int:
                 typ=0
                 while type(self.data1[typ])!=int:
                     typ+=1
                     if type(self.data1[typ])==str and self.data1[typ]=='1':
                         self.data1[typ]=1
                     elif type(self.data1[typ])==str and self.data1[typ]=='0':
                         self.data1[typ]=0
                     elif type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='+':
                         self.data1[typ]=np.random.randint(0,2,1)[0]
         if( action_bob == 4 ):
             self.bob_key.append(0)
             #self.bob_key=np.hstack((self.bob_key,0))
                 # reward = 0
             if( len(self.bob_key) == len(self.data2) ):
             # self.done = 
                 #self.bob_key=np.array(self.bob_key)
                 #print('This is data1 {} and data2 {} and Bob key {}'.format(self.data1,self.data2,self.bob_key))
                 print('This is the data1 before comparison {}'.format(self.data1))
                 self.data1=[self.data1]
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
         if( action_bob == 5 ):
             self.bob_key.append(1)
             #self.bob_key=np.hstack((self.bob_key,1))
             # reward = 0
             # If bob wrote enough bits
             if( len(self.bob_key) == len(self.data2) ):
                 # self.done = True
                 #self.bob_key=np.array(self.bob_key)
                 print('This is the data1 before comparison {}'.format(self.data1))
                 self.data1=[self.data1]
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
actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
LogicalStates=np.array([[1,0],[0,1]])
LogicalStates2bit=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
LogicalStates3bit=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
LogicalStates4bit=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
#statesColumns=[]
def qtableu(statesColumns):
    import pandas as pd
    actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
#    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
# =============================================================================
    #statesColumns.append(states)
    statesColumns=list(statesColumns)
    StateColumns=[]
    for i in range(len(statesColumns)):
        bob_keys=''.join(str(int(x)) for x in statesColumns[i][:len(statesColumns[i])])
        StateColumns.append(bob_keys)
        StatesColumns=StateColumns
    q=(len(StatesColumns),len(actions_list))
    Q=np.zeros(q)
    Qtable=pd.DataFrame(Q.T,columns=StatesColumns)
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
def Qlearning(inp,ma=False,qp=False,onebit=False,twobit=False,threebit=False,fourbit=False,gamma_v=0.001,learning_ra=1e-3):
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
    #sCol=[]
    for episode in range(episodes):
        import matplotlib.pyplot as plt
        gamma= gamma_v
        learning_rate=learning_ra 
        env=Qprotocol(8,inp,MultiAgent=ma,Qb=qp)
        state_n,inpu=env.reset(8)
        sCol=np.array(state_n[0])
        sCol=np.vstack(np.zeros((1,8)))
        Qtable=qtableu(sCol)
        #state_n=state_n[0]
        #state_n=str((state_n[0][0],state_n[0][1],state_n[0][2],state_n[0][3]))
        done=False
        reward_episode=[]
        steps=0
        reward=0
        while done!=True:
            z=[state_n[0][i]==sCol[-1][i] for i in range(len(state_n[0]))]
            z=np.array(z)
            print(z)
            if z.all()==True:
                q_val=''.join(str(int(x)) for x in state_n[0][:len(state_n[0])])
                #continue
                #Qtable=qtableu(sCol)
            else:
                sCol=np.vstack((sCol,state_n[0]))
                #print(sCol,sCol[-1])
                q_val=''.join(str(int(x)) for x in sCol[-1][:len(sCol[-1])])
                Qtable.insert(0,q_val,np.zeros(30))
                #df.insert(1, "newcol", [99, 99])
            steps+=1
            ravar=np.random.randint(len(actions_list), size=(1,len(actions_list)))/1000
            print('This is ravar {}'.format(ravar))
            print('This is q_val {}'.format(Qtable.loc[:,q_val]))
            Qtable = Qtable.T.groupby(level=0).first().T
            random_values=Qtable.loc[:,q_val] + ravar[0]#Qtable[str(state_n[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            print('This are random values {}'.format(random_values))
            actiona=np.argmax(random_values)
            action=np.array(actions_list[actiona])
            stat,reward,done,action_h,bob_key=env.step(action)
            q_val_n=''.join(str(int(x)) for x in stat[0][:len(stat[0])])#str((int(stat[0][0]),int(stat[0][1]),int(stat[0][2]),int(stat[0][3])))
            print('This is the sCol {}'.format(sCol))
            z=[stat[0][i]==sCol[-1][i] for i in range(len(stat[0]))]
            z=np.array(z)
            print(z)
            if z.all()==True:
                q_val=''.join(str(int(x)) for x in stat[0][:len(stat[0])])
                #continue
                #Qtable=qtableu(sCol)
            else:
                sCol=np.vstack((sCol,stat[0]))
                #print(sCol,sCol[-1])
                q_val=''.join(str(int(x)) for x in sCol[-1][:len(sCol[-1])])
                Qtable.insert(0,q_val,np.zeros(30))
                #Qtable=qtableu(sCol)
            #else:
            #    print('exist')
            print('These are the columns {}'.format(Qtable.columns))
            Qtable = Qtable.T.groupby(level=0).first().T
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
                inpus=str(inpu)
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

Q,r=Qlearning(1,onebit=True)