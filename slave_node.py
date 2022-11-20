from datetime import datetime
import random
import math
import csv
import socket
import numpy as np
import json
import os
import uuid
from env import env
from env import env_PPO
#notes:
'''
To Launch the agent, the config file must icnlude all the environment and agent parameters. Make sure master node is running first.
'''


#Name Node:
host_id = uuid.uuid4()

#Open config file:
with open("/data/sim/config.json","r") as file:
    config=json.load(file)

#Set number of simulation steps per simulation:

if config["NETWORK"] == "PPO":
    Ant = env_PPO(config,host_id)
else: 
    Ant = env(config,host_id)
Ant.launch()