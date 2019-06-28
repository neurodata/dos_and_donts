#%%

import graspy

import matplotlib.pyplot as plt
import numpy as np

from graspy.plot import heatmap
from graspy.simulations import sbm

def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels

m=100
n1 = [50, 50]
p1 = [[0.5, 0.2],
     [0.2, 0.05]]
dc1 = None #[i for i in range(50)]
# = sum(dc)
#dc = [d/s for d in dc]*2

np.random.seed(1)
P1 = []
for g in range(m):
    P1.append(sbm(n=n1, p=p1,dc=dc1))


lbls = _n_to_labels(np.array(n))
heatmap(P1[0],inner_hier_labels=lbls, title ='Graph from Population 1')

#%%
sig=0.1

p_delta = np.random.normal(loc=0,scale=sig,size = (n1[0],n1[0]))
for i in range(n1[0]):
    for j in range(i+1,n1[0]):
        p_delta[i,j] = p_delta[j,i]
p2 = np.zeros((n1[0]+1,n1[0]+1))
p2[:-1,:-1] = p_delta+0.5
p2[:,-1] = 0.2
p2[-1,:] = 0.2
p2[-1,-1] = 0.05

n2 = n1[0]*[1]+[50]
dc2=None

P2 = []
for g in range(m):
    P2.append(sbm(n=n2, p=p2,dc=dc2))

lbls = _n_to_labels(np.array(n))
heatmap(P2[0],inner_hier_labels=lbls, title ='Graph from Population 2')


#%%
