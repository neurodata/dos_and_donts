#%%

import graspy

import matplotlib.pyplot as plt
import numpy as np

from graspy.plot import heatmap
from graspy.simulations import sbm

n = [50, 50]
p = [[0.5, 0.2],
     [0.2, 0.05]]
dc = np.arange(100)/(np.sum(np.arange(100)))
np.random.seed(1)
G = sbm(n=n, p=p,dc=dc)



heatmap(G, title ='SBM Simulation')

#%%
