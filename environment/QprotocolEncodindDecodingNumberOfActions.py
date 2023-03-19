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
     def __init__(self,maxm,inp,MultiAgent=False):
         self.max_moves = maxm
         self.inpt=inp
         if MultiAgent==True:
             self.data1=inp
             self.data0=np.random.randint(0,2,len(inp))
             self.data2 = np.random.randint(0,2,len(inp))
         else:
             self.data0=np.random.randint(0,2,inp)
             self.data1 = np.random.randint(0,2,inp)
             self.data2 = np.random.randint(0,2,inp)
         # State for alice
         self.data0=self.data1
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
         print('this is the data1 {} and this is the type of data1 {}'.format(self.data1,type(self.data1)))
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
             if isinstance(self.data1, list)!=True:
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
             print('This is the data1 {} and type {} and encode'.format(self.data1,type(self.data1)))
             charsEn="X"
             if isinstance(self.data1, list)!=True:
                 self.data1=[self.data1]
                 if len(self.data1)==1 and type(self.data1[-1])!=str:# or type(self.data1)==int:
                     if self.data1==0:
                         self.data1='+'
                     else:
                         self.data1='-'
                 elif len(self.data1)==2 and type(self.data1[-1])!=str:# or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])==str:
                         typ+=1
                         if type(self.data1[typ])!=str and self.data1[typ]==0:
                             self.data1[typ]='+'
                         elif type(self.data1[typ])!=str and self.data1[typ]==1:
                             self.data1[typ]='-'
                         
                 elif len(self.data1)==3 and type(self.data1[-1])!=str:# or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])==str:
                         typ+=1
                         if type(self.data1[typ])!=str and self.data1[typ]==0:
                             self.data1[typ]='+'
                         elif type(self.data1[typ])!=str and self.data1[typ]==1:
                             self.data1[typ]='-'
                 elif len(self.data1)==4 and type(self.data1[-1])!=str:# or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])==str:
                         typ+=1
                         if type(self.data1[typ])!=str and self.data1[typ]==0:
                             self.data1[typ]='+'
                         elif type(self.data1[typ])!=str and self.data1[typ]==1:
                            self.data1[typ]='-'
         if( action_alice == 4 ):
             print('This is the data1 {} and type {} and encode'.format(self.data1,type(self.data1)))
             charsEn="Z"
             if isinstance(self.data1, list)!=True:
                 print('This is the type of the data {}'.format(type(self.data1)))
                 self.data1=[self.data1]
                 if len(self.data1)==1 and type(self.data1[-1])!=str: #or type(self.data1)==int:
                     if self.data1==0:
                         self.data1='0'
                     else:
                         self.data1='1'
                 elif len(self.data1)==2 and type(self.data1[-1])!=str: #or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])==str:
                         typ+=1
                         if type(self.data1[typ])!=str and self.data1[typ]==0:
                             self.data1[typ]='0'
                         elif type(self.data1[typ])!=str and self.data1[typ]==1:
                             self.data1[typ]='1'
                 elif len(self.data1)==3 and type(self.data1[-1])!=str:# or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])==str:
                         typ+=1
                         if type(self.data1[typ])!=str and self.data1[typ]==0:
                             self.data1[typ]='0'
                         elif type(self.data1[typ])!=str and self.data1[typ]==1:
                             self.data1[typ]='1'
                 elif len(self.data1)==4 and type(self.data1[-1])!=str:# or type(self.data1)!=int:
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
             if isinstance(self.data1, list)!=True:
                 self.data1=[self.data1]
                 if len(self.data1)==1 and type(self.data1[-1])==str or type(self.data1)!=int:
                     if self.data1=='-' or self.data1=='1':
                         self.data1=1
                     else:
                         self.data1=0
                 elif len(self.data1)==2 and type(self.data1[-1])==str or type(self.data1)!=int:
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
                 elif len(self.data1)==4 and type(self.data1[-1])==str or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])!=int:
                         typ+=1
                         if type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='1':
                             self.data1[typ]=1
                         if type(self.data1[typ])==str and self.data1[typ]=='+' or self.data1[typ]=='0':
                             self.data1[typ]=0
         if( action_bob == 3 ):
             charsEn="Z"
             if isinstance(self.data1, list)!=True:
                 self.data1=[self.data1]
                 if len(self.data1)==1 and type(self.data1[-1])==str or type(self.data1)==int:
                     if self.data1=='1':
                         self.data1=1
                     elif self.data1=='-' or self.data1=='+':
                         self.data1=np.random.randint(0,2,1)[0]
                     else:
                         self.data1=0
                 elif len(self.data1)==2 and type(self.data1[-1])==str: #or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])!=int:
                         typ+=1
                         if type(self.data1[typ])==str and self.data1[typ]=='1':
                             self.data1[typ]=1
                         elif type(self.data1[typ])==str and self.data1[typ]=='0':
                            self.data1[typ]=0
                         elif type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='+':
                            self.data1[typ]=np.random.randint(0,2,1)[0]
                 elif len(self.data1)==3 and type(self.data1[-1])==str:# or type(self.data1)!=int:
                     typ=0
                     while type(self.data1[typ])!=int:
                         typ+=1
                         if type(self.data1[typ])==str and self.data1[typ]=='1':
                             self.data1[typ]=1
                         elif type(self.data1[typ])==str and self.data1[typ]=='0':
                            self.data1[typ]=0
                         elif type(self.data1[typ])==str and self.data1[typ]=='-' or self.data1[typ]=='+':
                            self.data1[typ]=np.random.randint(0,2,1)[0]
                 elif len(self.data1)==4 and type(self.data1[-1])==str:# or type(self.data1)!=int:
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
             print(type(self.data1))
             if isinstance(self.data1, list)!=True:
                 self.data1=[self.data1]
             if (len(self.bob_key) == len(self.data1)) or type(self.data1)==int:
             # self.done = 
                 #self.bob_key=np.array(self.bob_key)
                 #print('This is data1 {} and data2 {} and Bob key {}'.format(self.data1,self.data2,self.bob_key))
                 print('This is the data1 before comparison {}'.format(self.data1))
                 a=[self.bob_key[i]==self.data0[i] for i in range(len(self.bob_key))]
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
             print(type(self.data1))
             if isinstance(self.data1, list)!=True:
                 self.data1=[self.data1]
             # If bob wrote enough bits
             if (len(self.bob_key) == len(self.data1)) or type(self.data1)==int:
                 # self.done = True
                 #self.bob_key=np.array(self.bob_key)
                 print('This is the data1 before comparison {}'.format(self.data1))
                 a=[self.bob_key[i]==self.data0[i] for i in range(len(self.bob_key))]
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
         if self.data0==[]:
             self.data0=np.random.randint(0,2,self.inpt)
             self.data1 = np.random.randint(0,2,self.inpt)
             self.data2 = np.random.randint(0,2,self.inpt)
         #if self.data0==[]:
         #    print('False')
         #    print('This is the data1 {}'.format(self.data1))
         #    self.data1=np.random.randint(0,2,self.inpt)
         #    self.data0=np.random.randint(0,2,self.inpt)
         else:
             self.data1=self.data1
             self.data2=self.data2
             self.data0=self.data0
         # State for alice
         #print('This is the type of data {}'.format(type(self.data1)))
         print('This is data1 {} and data0 {}'.format(self.data1,self.data0))
         if isinstance(self.data1, list)!=True:
             self.data1=[self.data1]
         else:
             self.data0=self.data1
         #if self.data0 in [0] or self.data0 in [1]:
         #   self.data1=self.data1(0,2,self.inpt)
         #   self.data0=self.data0(0,2,self.inpt)
         #print('this is the bitstring message {} and the target message {}'.format(self.data0,self.data1))
         
         z=[self.data1[i]==self.data0[i] for i in range(len([self.data1]))]
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
         return state,self.data0
     def render(self):
         print("---Alice---")
         print("- Datalog: ", self.alice_datalog)
         print("---Bob---")
         print("- Has Mail: ", self.bob_has_mail)
         print("- Datalog: ", self.bob_datalog)
         print("")
         return
