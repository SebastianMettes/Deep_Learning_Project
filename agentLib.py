import json
import os
import json
import torch
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


#Create network:

class MLP_agent(nn.Module):
    def __init__(self,agent_config):
        super().__init__()
        self.agent_config = agent_config
        self.HIDDEN_SIZE = agent_config['HIDDEN_SIZE']
        self.OBSERVE_SIZE = agent_config['OBSERVE_SIZE']
        self.N_ACTIONS = agent_config['N_ACTIONS']
        self.BATCH_SIZE = agent_config['BATCH_SIZE']
        self.PERCENTILE = agent_config['PERCENTILE']        
        self.agent_path = agent_config['agent_path']
        self.torque_multiplier = agent_config['torque_multiplier']
        self.version = 0
        self.cuda_test = agent_config['cuda']
        self.initialize()
        if os.path.isfile(os.path.join(self.agent_path,str(1),'model.pt')):
            self.update_version()

    def initialize(self):
        obs_size = self.OBSERVE_SIZE
        hidden_size = self.HIDDEN_SIZE
        n_actions = self.N_ACTIONS

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,n_actions),
            nn.ReLU(),
            nn.Linear(n_actions,n_actions),
            nn.ReLU(),
            nn.Linear(n_actions,n_actions),
        )
        if self.cuda_test == True:
            self.cuda()
        else:
            self.cpu()


    def cuda(self):
        self.net = self.net.cuda()
    def cpu(self):
        self.net = self.net.cpu()

    def forward(self,observation):
        return self.net(observation)

    def calc_action(self,agent_version,state,torque_input):
        if self.version!=agent_version:
            self.net.load_model(os.path.join(self.agent_path,str(agent_version),'model.pt'))
        action_probability = self.net.forward(state.float().unsqueeze(0))
        action_probability = nn.functional.softmax(action_probability,dim=1)
        action_probability = action_probability.data.numpy()[0]
        action = np.random.choice(len(action_probability),p=action_probability) #return an action based on softmax probability that action is most likely to produce positive reward
        
        #Create output options array:
        a0 = np.array([-1,1])
        a1 = np.array([-1,1])
        a2 = np.array([-1,1])
        a3 = np.array([-1,1])
        a4 = np.array([-1,1])
        a5 = np.array([-1,1])
        a6 = np.array([-1,1])
        a7 = np.array([-1,1])
        comb_array = np.array(np.meshgrid(a0,a1,a2,a3,a4,a5,a6,a7)).T.reshape(-1,8)
        output = torque_input+comb_array[action,:]*self.torque_multiplier
        return output,action

    def load_model(self,version_path):
        self.net.load_state_dict(torch.load(version_path))

    def save_model(self,directory_path,optimizer):
        tmp_name = directory_path+'_tmp'
        if os.path.isdir(tmp_name) == False:        
            os.mkdir(tmp_name, mode = 0o777)

        torch.save(self.net.state_dict(),os.path.join(tmp_name,'model.pt'))
        torch.save(optimizer.state_dict(),os.path.join(tmp_name,'optimizer.pt'))
        shutil.move(tmp_name,directory_path)

    def update_version(self):
        i = 1
        while os.path.isdir(os.path.join(self.agent_path,str(i))):
            i+=1
        i-=1
        if i>self.version:
            time.sleep(0.1)
            print('loading version ',i)
            self.load_model(os.path.join(self.agent_path,str(i),'model.pt'))
            self.version = i
        return(i)





#