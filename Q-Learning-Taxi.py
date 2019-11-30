import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env=gym.make("Taxi-v3").env

#Q Table
q_table=np.zeros([env.observation_space.n,env.action_space.n])
#Hyperparameter
alpha=0.1
gamma=0.9
epsilon=0.1
#Plotting Metrix
reward_list=[]
dropout_list=[]

episode=100000
for i in range(1,episode):
    state=env.reset()
    reward_count=0
    dropout_count=0
    while True:
        
        if random.uniform(0,1) < epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(q_table[state])
        
        next_state, reward, done, info = env.step(action)
        
        old_value=q_table[state,action]
        next_max=np.max(q_table[next_state])
        
        new_value=(1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        q_table[state,action]=new_value
        
        state=next_state
        
        if reward == -10:
            dropout_count +=1
            
        reward_count +=reward
        if done:
            break
    if i%10==0:
        dropout_list.append(dropout_count)
        reward_list.append(reward_count)
        print("Episode{},reward{},wrong dropout{}".format(i,reward_count,dropout_count))

#%%
fig,axs=plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropout")

axs[0].grid(True)
axs[1].grid(True)

plt.show()
#%%
env.s=env.encode(0,4,4,1)
env.render()





















