import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm.auto import tqdm


class DL_Utilities():
    def __init__(self,samples=False,epochs=False):
        self.data = np.genfromtxt("/data/sim/agent/data.csv",delimiter=",")
        self.trial = "/data/sim/trial/"
        self.agent = "/data/sim/agent/"
        with open("/data/sim/config.json","r") as file:
            config=json.load(file)

        self.obs_size = config["OBSERVE_SIZE"]
        self.obs_per_episode = config["BATCH_SIZE"]*self.obs_size
        self.num_steps = config["num_steps"]

        if samples:
            self.samples = samples
        else:
            self.samples = [10,20,30,40,50,60,70,80,90,100] #Samples used for course plotting
        if epochs:
            self.epochs = epochs
        else:
            self.epochs = [1,100,200,300,400,500]
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
        self.data_matrix = np.zeros([len(self.epochs),len(self.samples),(self.num_steps),(self.obs_size)])
        for i in tqdm(range(len(self.epochs))):
            episode_list = os.listdir(os.path.join(self.trial,str(self.epochs[i])))
            for j in range(len(self.samples)):
                with open(os.path.join(self.trial,str(self.epochs[i]),str(episode_list[j]))) as file:
                    data = json.load(file)
                    for k in range(self.num_steps):
                        self.data_matrix[i,j,k,:] = data[k][1]   #i = epoch number, j = episode number, k = step number
        
    def position(self):
        for i in range(len(self.epochs)):
            x_data = self.data_matrix[i,:,:,0]
            y_data = self.data_matrix[i,:,:,1]
            plt.scatter(x_data,y_data)
        plt.show()
        


    def velocity(self):
        for i in range(len(self.epochs)):
            x_data = self.data_matrix[i,:,:,15]
            y_data = self.data_matrix[i,:,:,16]
            plt.scatter(x_data,y_data)
        plt.show()
        

    def joint_angle_plot(self,joint_num = 0):
        pass


if __name__ == "__main__":
    plots = DL_Utilities()
    plots.collate_data()
    plots.position()
    plots.velocity()
    plots.loss()
    plots.reward()
    plots.position()