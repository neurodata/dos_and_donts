#%%

import graspy

import matplotlib.pyplot as plt
import numpy as np

from graspy.plot import heatmap
from graspy.simulations import sbm
from scipy.stats import chisquare
from scipy.stats import fisher_exact

def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels

#Create population 1
m=100
n1 = [50, 50]
p1 = [[0.5, 0.2],
     [0.2, 0.05]]
dc1 = None #[i for i in range(50)]
# = sum(dc)
#dc = [d/s for d in dc]*2

np.random.seed(1)
for g in range(m):
        if g==0:
                P1 = sbm(n=n1, p=p1,dc=dc1)
                P1 = np.expand_dims(P1,2)
        else:
                P1 = np.append(P1,np.expand_dims(sbm(n=n1, p=p1,dc=dc1),2),axis=2)

lbls1 = _n_to_labels(np.array(n1))
#heatmap(P1[:,:,0],inner_hier_labels=lbls1, title ='Graph from Population 1')

#%%
#Create population 2
sig=0.0

p_delta = np.random.normal(loc=0,scale=sig,size = (n1[0],n1[0]))
np.fill_diagonal(p_delta,0)
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


for g in range(m):
        if g==0:
                P2 = sbm(n=n2, p=p2,dc=dc2)
                P2 = np.expand_dims(P2,2)
        else:
                P2 = np.append(P2,np.expand_dims(sbm(n=n2, p=p2,dc=dc2),2),axis=2)

lbls2 = _n_to_labels(np.array(n2))
#heatmap(P2[:,:,0],inner_hier_labels=lbls2, title ='Graph from Population 2')

#%%
#Edgewise test
log_p = np.zeros(P1.shape[:2])
for i in range(P1.shape[0]):
        for j in range(i+1,P1.shape[1]):
                edges_1 = P1[i,j,:]
                edges_2 = P2[i,j,:]
                table = np.array([[np.sum(edges_1),np.sum(edges_1 == 0)],
                        [np.sum(edges_2),np.sum(edges_2 == 0)]])
                _,p = fisher_exact(table)
                log_p[i,j] = np.log(p)
                log_p[j,i] = np.log(p)

num_tests = P1.shape[0]*(P1.shape[0]-1)/2
edgewise_sig = np.sum(log_p < np.log(0.05/num_tests))
print("Number of significant edges from Fisher's exact with a=0.05, Bonferroni Correction: " + str(edgewise_sig))

#heatmap(log_p,inner_hier_labels=lbls1, title ='Log-p for Edgewise Fisher Exact')

#%%
#Blockwise test
indices_1 = np.cumsum(n1)
num_blocks = indices_1.shape[0]
log_p_blocks = np.zeros((num_blocks,num_blocks))
lbls_block = np.arange(0,num_blocks)
for i in np.arange(num_blocks):
        if i==0:
                start_i = 0
        else:
                start_i = indices_1[i-1]
        end_i = indices_1[i]
        for j in np.arange(i,num_blocks):
                if i==0:
                        start_j = 0
                else:
                        start_j = indices_1[j-1]
                end_j = indices_1[j]
                
                edges_1 = P1[start_i:end_i,start_j:end_j,:].flatten()
                edges_2 = P2[start_i:end_i,start_j:end_j,:].flatten()
                table = np.array([[np.sum(edges_1),np.sum(edges_1 == 0)],
                        [np.sum(edges_2),np.sum(edges_2 == 0)]])
                _,p = fisher_exact(table)
                log_p_blocks[i,j] = np.log(p)
                log_p_blocks[j,i] = np.log(p)

num_tests = num_blocks*(num_blocks+1)/2
edgewise_sig = np.sum(log_p_blocks < np.log(0.05/num_tests))
print("Number of significant blocks from Fisher's exact with a=0.05, Bonferroni Correction: " + str(edgewise_sig))

heatmap(log_p_blocks,inner_hier_labels=lbls_block, title ='Log-p for Edgewise Fisher Exact')



#%%
