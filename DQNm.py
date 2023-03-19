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
import environment.ClassicalCommunicationChNumberOfActions as Cch
import environment.QprotocolEncodindDecodingNumberOfActions as Qch
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




EPISODES=36
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
    state_size=8
    actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
#    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
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
        qpO=Qch.Qprotocol(8,inpu,MultiAgent=ma)
        state_n,inp=qpO.reset(8)
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

def Dqnsimulation(inpu,ag,ma):
    batch_size=24
    EPISODES=100
    solved=0
    steps_epi=[]
    qval=[]
    qval_pr=[]
    total_episodes=[]
    cumre=0
    actionDecision=[]
    cumulative_reward=[]
    total_fidelity=[]
    reward_episode=[]
    for e in range(EPISODES):
            env=Qch.Qprotocol(8,inpu,MultiAgent=ma)
            state_n,inp=env.reset(8)
            steps_ep=0
            done=False
            state = np.array(state_n[0])
            state=np.reshape(state, [1, state_size])
            while done!=True:
                actiona=ag.act(state)
                actiona=np.array(actiona)
                actionA = actions_list[actiona]
                actionDecision.append(actionA)
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
    return results,actionDecision

def actli(actis):
    thd=[]
    actis=np.array(actis)
    for i in range(0,len(actis)):
        thd.append([actis[i][0],actis[i][1],i])
    thd=np.array(thd)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(thd[:,0], thd[:,1], thd[:,2])
    plt.show()



state_size=4
state_size=8
actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
action_size=len(actions_list)#env.action_space.n

agent = DQNAgent(state_size, action_size)
agent,r=Dqn(1,agent,False,False)
print(r,file=open('randomOneBitDqnTraining.txt','w'))
r,actd=Dqnsimulation(1,agent,False)
actli(actd)
print(r,file=open('randomOneBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(2,agent,False,False)
print(r,file=open('randomTwoBitDqnTraining.txt','w'))
r,actd=Dqnsimulation(2,agent,False,)
actli(actd)
print(r,file=open('randomTwoBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(3,agent,False,False)
print(r,file=open('randomThreeBitDqnTraining.txt','w'))
r,actd=Dqnsimulation(3,agent,False)
actli(actd)
print(r,file=open('randomThreeBitDqnTesting.txt','w'))
agent = DQNAgent(state_size, action_size)
agent,r=Dqn(4,agent,False,False)
print(r,file=open('randomFourBitDqnTraining.txt','w'))
r,actd=Dqnsimulation(4,agent,False)
actli(actd)
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
