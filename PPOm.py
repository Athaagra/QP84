#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:22:04 2023

@author: Optimus
"""
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal 
import sys
import environment.ClassicalCommunicationChNumberOfActions as Cch
import environment.QprotocolEncodindDecodingNumberOfActions as Qch
#temp = sys.stdout                 # store original stdout object for later
#sys.stdout = open('log.txt', 'w')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
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
    #print('This is the logprobabilities all {}'.format(logprobabilities_all))
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability

@tf.function
def sample_action(observation,md):
    logits = md(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer,md
):
    
    #print('This is the train policy')
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            #(observation_buffer)
            logprobabilities(md(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        #print('This is the ratio {} advantage buffer'.format(ratio))
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        #print('This is the min_advantage {} 1+ clip_ratio*advantage_buffer, 1-clip_ratio*advantage_buffer '.format(min_advantage))
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
        #ld=np.array(policy_loss)
        #fp = open('NN_for_Dymola.txt', 'w' ,encoding='UTF-8')
        #tf.print(policy_loss, output_stream=sys.stdout)
        #tf.io.write_file('Training.txt',policy_loss)
    policy_grads = tape.gradient(policy_loss, md.trainable_variables)
    #print('Policy grads {}'.format(policy_grads))
    policy_optimizer.apply_gradients(zip(policy_grads, md.trainable_variables))
    #print('policy optimizer {}'.format(policy_optimizer))
    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(md(observation_buffer), action_buffer)
    )
    #print('This is the kl {} logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer)'.format(kl))
    kl = tf.reduce_sum(kl)
    return kl#,policy_loss


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer,cc):
    #print('This is the observation_buffer {} and the return buffer {}'.format(observation_buffer, return_buffer))
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        #print('This is the tape {}'.format(tape))
        value_loss = tf.reduce_mean((return_buffer - cc(observation_buffer)) ** 2)
        #ldd=value_loss
        #tf.print(value_loss, output_stream=sys.stdout)#print('This is the value loss {}'.format(ldd))
    value_grads = tape.gradient(value_loss, cc.trainable_variables)
    #print('This is the value_grads {} value_loss, critic.trainbable_variables'.format(value_grads))
    value_optimizer.apply_gradients(zip(value_grads, cc.trainable_variables))
    #return value_loss

# Hyperparameters of the PPO algorithm
gamma = 0.001
clip_ratio = 0.01
policy_learning_rate = 1e-11
value_function_learning_rate = 1e-11
train_policy_iterations = 200
train_value_iterations = 200
lam = 0.97
target_kl = 0.01
hidden_sizes = (16, 16)

# True if you want to render the environment
render = False

# Initialize the environment and get the dimensionality of the
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
# Iterate over the number of epochs
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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 13))
    plt.bar(1,pvalue)
    plt.xlabel(f'Mannwhitney Test')
    plt.ylabel('Probability')
    plt.title(str(resultss))#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return resultss
def proximalpo(inpu,ac,cr,ma):
    epochs = 100
    steps_ep=[]
    q_value_critic=[]
    action_actor=[]
    cum=[]
    total_episodes=[]
    cumre=0
    total_fidelity=[]
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done=False
        env=Qch.Qprotocol(8,inpu,MultiAgent=ma)
        state,inp=env.reset(8)
        actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
        #actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        observation = state[0]
        episode_return=0
        episode_length=0
        while done != True:
        # Iterate over the steps of each epoch
            if render:
                print(render)
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=np.array(actiona)
            log=np.array(logits[0])
            actiona=actiona[0]
            action_actor.append(log[actiona])
            action=np.array(actions_list[actiona])
            new_state,reward,done,info,bob_key=env.step(action)
            episode_return += reward
            episode_length += 1        
            # Get the value and log-probability of the action
            value_t = cr(observation)
            q_value_critic.append(value_t)
            logprobability_t = logprobabilities(logits, actiona)
            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, actiona, reward, value_t, logprobability_t)
            # Update the observation
            observation = np.array(new_state[0])
            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal: #or (t == steps_per_epoch - 1):
                last_value = 0 if done else cr(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                steps_ep.append(episode_length)
                num_episodes += 1
                cumre+=reward
                cum.append(cumre)
                state,inp=env.reset(4)
                observation=np.array(state[0]) 
                episode_return=0
                episode_length = 0
                if reward==1:
                    total_episodes.append(1)
                else:
                    total_episodes.append(0)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()    
        # Update the policy and implement early stopping using KL divergence
        locy=[]
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, ac
            )
            #locy.append(lc)
            if kl > 1.5 * target_kl:
                # Early Stopping
                break
    #    print('This is the kl {}'.format(kl))
        # Update the value function
        cry=[]
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer,cr)

        rewards_during_training.append(sum_return / num_episodes)
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    def save_weights(ac,cr,inpt,maxm):
        path= '/home/Optimus/Desktop/QuantumComputingThesis/'
        ac.save(path+ '_actor'+str(maxm)+'One'+str(inpt)+'.h5')
        cr.save(path+ '_critic'+str(maxm)+'One'+str(inpt)+'.h5')
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy:{}'.format(sum(total_episodes)))#.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cum)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Cumulative:{}'.format(max(cum)))#.format(solved/episodes))
    plt.savefig('cumulativeppo.png')
    plt.show()
    total_fidelity=[ abs(sum(i))**2 if type(i)!=int else i for i in total_fidelity]
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(q_value_critic)
    plt.plot(action_actor)
    plt.xlabel(f'loss of episode')
    plt.ylabel('Q-value')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation '+str(inpu)+' and the Q-value of Proximal Policy Evaluation')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_ep)
    plt.xlabel(f'Number of steps of each episode')
    plt.ylabel('Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation and the number of steps '+str(inpu)+' of Proximal Policy Evaluation {}'.format(np.mean(steps_ep)))
    plt.show()
    save_weights(ac,cr,inpu,4)
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(count),'Cumulative:'+str(max(cum)),'Steps:'+str(np.mean(steps_ep))])
    return actor,critic,results

def pposimulation(inpu,ac,ma):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    state_size=8
    actions_made_list=[]
    actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
    #actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        env=Qch.Qprotocol(8,inpu,MultiAgent=ma)
        state,inp=env.reset(8)
        # reset environment and episode reward
        print('This is input {}'.format(inp))
        ep_reward = 0
        done=False
        observation = state[0]
        steps_ep=0
        while done!=True:
            print('This is the episode {}'.format(i_episode))
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=actiona[0]
            actiona=np.array(actions_list[actiona])
            actions_made_list.append(actiona)
            stat,reward,done,action_h,bob_keya=env.step(actiona)
            observation=stat[0]
            steps_ep+=1
            if done==True:
                bob_key=bob_keya
                steps_epi.append(steps_ep)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward==1:
                    r=1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    solved+=1
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    solved+=0
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Cumulative:{}'.format((cumulative_reward[-1])))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    total_fidelity=[ abs(sum(i))**2 if type(i)!=int else i for i in total_fidelity]
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Steps')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results,actions_made_list


# Hyperparameters of the PPO algorithm
gamma = 0.001
clip_ratio = 0.01
policy_learning_rate = 1e-11
value_function_learning_rate = 1e-11
train_policy_iterations = 200
train_value_iterations = 200
lam = 0.97
target_kl = 0.01
hidden_sizes = (16, 16)

# True if you want to render the environment
render = False

# Initialize the environment and get the dimensionality of the
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
# Iterate over the number of epochs
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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 13))
    plt.bar(1,pvalue)
    plt.xlabel(f'Mannwhitney Test')
    plt.ylabel('Probability')
    plt.title(str(resultss))#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return resultss
def proximalpo(inpu,ac,cr,ma):
    epochs = 100
    steps_ep=[]
    q_value_critic=[]
    action_actor=[]
    cum=[]
    total_episodes=[]
    cumre=0
    total_fidelity=[]
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done=False
        env=Qch.Qprotocol(8,inpu,MultiAgent=ma)
        state,inp=env.reset(8)
        actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
        #actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        observation = state[0]
        episode_return=0
        episode_length=0
        while done != True:
        # Iterate over the steps of each epoch
            if render:
                print(render)
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=np.array(actiona)
            log=np.array(logits[0])
            actiona=actiona[0]
            action_actor.append(log[actiona])
            action=np.array(actions_list[actiona])
            new_state,reward,done,info,bob_key=env.step(action)
            episode_return += reward
            episode_length += 1        
            # Get the value and log-probability of the action
            value_t = cr(observation)
            q_value_critic.append(value_t)
            logprobability_t = logprobabilities(logits, actiona)
            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, actiona, reward, value_t, logprobability_t)
            # Update the observation
            observation = np.array(new_state[0])
            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal: #or (t == steps_per_epoch - 1):
                last_value = 0 if done else cr(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                steps_ep.append(episode_length)
                num_episodes += 1
                cumre+=reward
                cum.append(cumre)
                state,inp=env.reset(4)
                observation=np.array(state[0]) 
                episode_return=0
                episode_length = 0
                if reward==1:
                    total_episodes.append(1)
                else:
                    total_episodes.append(0)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()    
        # Update the policy and implement early stopping using KL divergence
        locy=[]
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, ac
            )
            #locy.append(lc)
            if kl > 1.5 * target_kl:
                # Early Stopping
                break
    #    print('This is the kl {}'.format(kl))
        # Update the value function
        cry=[]
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer,cr)

        rewards_during_training.append(sum_return / num_episodes)
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    def save_weights(ac,cr,inpt,maxm):
        path= '/home/Optimus/Desktop/QuantumComputingThesis/'
        ac.save(path+ '_actor'+str(maxm)+'One'+str(inpt)+'.h5')
        cr.save(path+ '_critic'+str(maxm)+'One'+str(inpt)+'.h5')
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy:{}'.format(sum(total_episodes)))#.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cum)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Cumulative:{}'.format(max(cum)))#.format(solved/episodes))
    plt.savefig('cumulativeppo.png')
    plt.show()
    total_fidelity=[ abs(sum(i))**2 if type(i)!=int else i for i in total_fidelity]
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(q_value_critic)
    plt.plot(action_actor)
    plt.xlabel(f'loss of episode')
    plt.ylabel('Q-value')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation '+str(inpu)+' and the Q-value of Proximal Policy Evaluation')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_ep)
    plt.xlabel(f'Number of steps of each episode')
    plt.ylabel('Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation and the number of steps '+str(inpu)+' of Proximal Policy Evaluation {}'.format(np.mean(steps_ep)))
    plt.show()
    save_weights(ac,cr,inpu,4)
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(count),'Cumulative:'+str(max(cum)),'Steps:'+str(np.mean(steps_ep))])
    return actor,critic,results

def pposimulation(inpu,ac,ma):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    state_size=8
    actions_made_list=[]
    actions_list=[(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(2,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,2),(2,3),(2,4),(2,5),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,0),(4,1),(4,2),(4,3),(4,4),(4,5)]
    #actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        env=Qch.Qprotocol(8,inpu,MultiAgent=ma)
        state,inp=env.reset(8)
        # reset environment and episode reward
        print('This is input {}'.format(inp))
        ep_reward = 0
        done=False
        observation = state[0]
        steps_ep=0
        while done!=True:
            print('This is the episode {}'.format(i_episode))
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=actiona[0]
            actiona=np.array(actions_list[actiona])
            actions_made_list.append(actiona)
            stat,reward,done,action_h,bob_keya=env.step(actiona)
            observation=stat[0]
            steps_ep+=1
            if done==True:
                bob_key=bob_keya
                steps_epi.append(steps_ep)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward==1:
                    r=1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    solved+=1
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    solved+=0
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Cumulative:{}'.format((cumulative_reward[-1])))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    total_fidelity=[ abs(sum(i))**2 if type(i)!=int else i for i in total_fidelity]
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Steps')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results,actions_made_list
observation_dimensions = 8
num_actions = 30
steps_per_epoch=15
# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

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
    
# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rewards_during_training=[]
value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(1,actor,critic,False)
print(r,file=open('randomOneBitPPOTraining.txt','w'))
r,sav_actions=pposimulation(1, actor,False)
actli(sav_actions)
print(r,file=open('randomOneBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(2,actor,critic,False)
#sys.stdout.close()                # ordinary file object
#sys.stdout = temp 
print(r,file=open('randomTwoBitPPOTraining.txt','w'))
r,sav_actions=pposimulation(2, actor,False)
actli(sav_actions)
print(r,file=open('randomTwoBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(3,actor,critic,False)
print(r,file=open('randomThreeBitPPOTraining.txt','w'))
r,sav_actions=pposimulation(3, actor,False)
actli(sav_actions)
print(r,file=open('randomtThreeBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(4,actor,critic,False)
print(r,file=open('randomFourBitPPOraining.txt','w'))
r,sav_actions=pposimulation(4, actor,False)
actli(sav_actions)
print(r,file=open('randomFourBitPPOTesting.txt','w'))
