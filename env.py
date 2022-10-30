import gymnasium as gym
from agentLib import MLP_agent
import datetime
import os
import json
from datetime import datetime
import torch

class env():
    def __init__(self,config,host_id):
        self.config = config
        self.num_steps = config['num_steps']
        self.host_id = host_id

    def launch(self):
        self.session = gym.make('Ant-v4')
        agent = MLP_agent(self.config)


        while True:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d%H%M%S")

            agent_version = agent.update_version()
            filename = str(self.host_id)+"."+date_time+".JSON"
            filename = os.path.join(self.config["save_dir"],str(agent_version),filename)

            observation, info = self.session.reset()
            observation = torch.tensor(observation)
            action = [0,0,0,0,0,0,0,0] #Initialize 0-action as starting condition            

            state_tensor = []

            for i in range(self.num_steps):
                action,digit = agent.calc_action(agent_version,observation,action)
                observation_new, reward, terminated, truncated, info = self.session.step(action)
                state_tensor.append((observation.tolist(),observation_new.tolist(),digit,reward))

                with open(filename,"w") as file:
                    file.write(json.dumps(state_tensor,indent=0))
                observation_new = torch.tensor(observation_new)
                observation = observation_new
                if terminated or truncated:
                    observation, info = self.session.reset()
                    break