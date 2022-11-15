Jason: To implement a RNN, you should be able to create a new agent in agentLib.py. As long as all the inputs / outputs are accounted for - you shouldn't have an issue training.

Dong / Sebastian: To setup a an actor-critic / DPPO training pipeline, you'll likely need to change how episodes are selected for training. The actor should fit, I think, fit in the agentLib.py format, whereas the critic will likely require modification to the "master_node.py", which controls how the new networks get trained down / which episodes are selected. If you make a new master_node.py - make a new file / name it something like master_node_DPPO.py . 






