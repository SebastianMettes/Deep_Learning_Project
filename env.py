import gymnasium as gym
from agentLib import MLP_agent
import datetime
import os
import json

class env():
    def __init__(self,config,host_id):
        self.config = config
        self.num_steps = config['num_steps']
        self.host_id = host_id

    def launch(self,cuda):
        self.session = gym.make('Ant-v4',render_mode="human")
        agent = MLP_agent(self.config)
        if cuda == True:
            agent.cuda()
        else:
            agent.cpu()


        while True:
            now = datetime.now()
            date_time = now.strftime("%Y%m%d%H%M%S")

            agent_version = agent.update_version()
            filename = str(self.host_id)+"."+date_time+".JSON"
            filename = os.path.join(self.config["save_dir"],str(agent_version),filename)

            observation, info = env.reset()
            action = [0,0,0,0,0,0,0,0] #Initialize 0-action as starting condition            

            state_tensor = []

            for i in range(self.num_steps):
                action,digit = MLP_agent.calc_action(observation,action)
                observation_new, reward, terminated, truncated, info = env.step(action)
                state_tensor.append((observation,observation_new,digit,reward))
                with open(filename,"w") as file:
                    file.write(json.dumps(state_tensor,indent=0))

                if terminated or truncated:
                    observation, info = env.reset()
                    break