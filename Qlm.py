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

#import environment.ClassicalCommunicationChNumberOfActions as Cch
import environment.QprotocolEncodindDecodingNumberOfActions as Qch
import random
import matplotlib.pyplot as plt
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
    actions_list_made=[]
    #sCol=[]
    for episode in range(episodes):
        import matplotlib.pyplot as plt
        gamma= gamma_v
        learning_rate=learning_ra 
        env=Qch.Qprotocol(8,inp,MultiAgent=ma)
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
            #print('This is ravar {}'.format(ravar))
            #print('This is q_val {}'.format(Qtable.loc[:,q_val]))
            Qtable = Qtable.T.groupby(level=0).first().T
            random_values=Qtable.loc[:,q_val] + ravar[0]#Qtable[str(state_n[0])] + np.random.randint(11, size=(1,len(actions_list)))/1000
            #print('This are random values {}'.format(random_values))
            actiona=np.argmax(random_values)
            action=np.array(actions_list[actiona])
            actions_list_made.append(action)
            stat,reward,done,action_h,bob_key=env.step(action)
            q_val_n=''.join(str(int(x)) for x in stat[0][:len(stat[0])])#str((int(stat[0][0]),int(stat[0][1]),int(stat[0][2]),int(stat[0][3])))
            #print('This is the sCol {}'.format(sCol))
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
            #print('These are the columns {}'.format(Qtable.columns))
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
                    #print('This is the tp {} and the length of the tp {}'.format(tp[0],len(tp[0])))
                    inpus=str(inpu)
                    Fidelity=abs(sum(tp))**2
                elif twobit==True and len(bob_key)==2:
                    inpus=''.join(str(x) for x in inpu[0])
                    #print('This is the inpus {}'.format(inpus))
                    bob_keys=''.join(str(x) for x in bob_key[:2])#len(inpu)])
                    print('This is the bob_key {}'.format(bob_key))
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                elif threebit==True and len(bob_key)==3:
                    inpus=''.join(str(x) for x in inpu)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inpu)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                elif fourbit==True and len(bob_key)==4:
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
    #print('This is the total fidelity {}'.format(total_fidelity))
    total_fidelity=[ abs(sum(i))**2 if type(i)!=int else i for i in total_fidelity]
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
    return Qtable,results,actions_list_made

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
#Q,r,sav_actions=Qlearning(1,onebit=True)
#actli(sav_actions)
#Q,r,sav_actions=Qlearning(2,twobit=True)
#actli(sav_actions)
Q,r,sav_actions=Qlearning(3,threebit=True)
actli(sav_actions)
Q,r,sav_actions=Qlearning(4,fourbit=True)
actli(sav_actions)
