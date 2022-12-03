import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm.auto import tqdm

#data = [self.path_summary,self.action_summary,self.reward_summary,self.actor_loss_summary,self.test_iteration]

title = 'PPO + Lstm. Network 500,000 steps Results'
path = '/home/smettes/code/DLP/Deep_Learning_Project/PPO/backup/500_000_steps_lstm_LR=3E-4Hid=128'
file = open(''.join([path,'/graphable_results.json']))

path_summary,action_summary,reward_summary,actor_loss_summary,test_iteration = json.load(file)

#Plot paths:
total_episodes = len(path_summary)
first = np.array(path_summary[round(total_episodes/4)][:])
second = np.array(path_summary[round(total_episodes/4)*2][:])
third = np.array(path_summary[round(total_episodes/4)*3][:])
final = np.array(path_summary[-1][:])
fig = plt
fig.figure(1)
fig.plot(first[:,0],first[:,1],label = '25% Complete')
fig.plot(second[:,0],first[:,1],label = '50% Complete')
fig.plot(third[:,0],first[:,1],label = '75% Complete')
fig.plot(final[:,0],first[:,1],label = '100% Complete')
fig.title(''.join([title,'--paths']))
fig.legend()

fig.savefig(''.join([path,'/path']))

loss = np.array(actor_loss_summary)
loss = loss.astype(float)
reward = np.array(reward_summary)
reward = reward.astype(float)
iteration = np.array(test_iteration)
iteration = iteration.astype(float)
fig.figure(2)
fig.plot(iteration[1:],loss)
fig.title(''.join([title,'--loss']))
fig.xlabel('iteration #')
fig.savefig(''.join([path,'/loss']))

fig.figure(3)
fig.plot(iteration[1:],reward)
fig.title(''.join([title,'--reward']))
fig.xlabel('iteration #')


fig.savefig(''.join([path,'/reward']))
