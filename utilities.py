import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm.auto import tqdm


class DL_Utilities():
    def __init__(self):
        self.data = np.genfromtxt("/data/sim/agent/data.csv",delimiter=",")
        self.trial = "/data/sim/trial"
        self.agent = "/data/sim/agent"
        self.samples = [10,20,30,40,50,60,70,80,90,100] #Samples used for course plotting

    def loss(self):
        plt.plot(self.data[:,0],self.data[:,2])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss Value')
        plt.show()

    def reward(self):
        plt.plot(self.data[:,0],self.data[:,1])
        plt.xlabel('Epoch Number')
        plt.ylabel('Reward Value')
        plt.show()

    def collate_data(self):
        pass

    def position_plot(self):
        pass

    def velocity_plot(self):
        pass

    def joint_angle_plot(self,joint_num = 0)


if __name__ == "__main__":
    plots = DL_Utilities()
    plots.loss()
    plots.reward()