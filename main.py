import gymnasium as gym
from agent import agent

num_steps = 1000

#Launch the environment
env = gym.make('Ant-v4',render_mode="human")
observation, info = env.reset()

#Initialize the agent
for i in range(num_steps):
    #Future implementation:
    #action = agent.action(observation)
    action = env.action_space.sample() #random input
    observation, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        observation, info = env.reset()

env.close()



#