# Deep_Learning_Project
CD 7643 Deep Learning Project

python version = 3.8
pytorch installed with CUDA 11.6

To run in ubuntu:
install gymnasium[mujoco] by running: pip install gymnasium[mujoco]

Once running, setup the following folders to store data and models on the C drive:

\data\sim\trial
\data\sim\agent
\data\sim\difficult

in the \data\sim folder, drop the config.json file. Adjust all parameters here. Note - "BATCH_SIZE" is the number of trials to be run before optimizing. "gpu_batch_size" is the max number of trials to pass in per optimization batch. "gpu_batch_size" must be less than BATCH_SIZE*(1-PERCENTILE/100).
