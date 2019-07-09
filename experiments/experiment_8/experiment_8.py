#%%
import graspy

import matplotlib.pyplot as plt
import numpy as np

from graspy.plot import heatmap
from graspy.simulations import sbm
from scipy.stats import chisquare
from scipy.stats import fisher_exact


def _n_to_labels(n):
        n = np.array(n)
        n_cumsum = n.cumsum()
        labels = np.zeros(n.sum(), dtype=np.int64)
        for i in range(1, len(n)):
                labels[n_cumsum[i - 1] : n_cumsum[i]] = i
        return labels

#Create population 1
print('Making population 1...')
m=100
n1 = [50, 50]
p1 = [[0.5, 0.2],
     [0.2, 0.05]]
dc1 = None #[i for i in range(50)]
# = sum(dc)
#dc = [d/s for d in dc]*2

np.random.seed(0)
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
print('Making population 2...')
sig=0.03
print('Sigma is ' + str(sig))

p_delta = np.random.normal(loc=0,scale=sig,size = (n1[0],n1[0]))
p_delta[p_delta<-0.5] = -0.5
p_delta[p_delta>0.5] = 0.5
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

lbls2 = _n_to_labels(n1)
#heatmap(P2[:,:,0],inner_hier_labels=lbls2, title ='Graph from Population 2')

#%%
#Edgewise test
print('Performing edgewise test...')
ps = np.zeros(P1.shape[:2])
for i in range(P1.shape[0]):
	for j in range(i+1,P1.shape[1]):
		edges_1 = P1[i,j,:]
		edges_2 = P2[i,j,:]
		table = np.array([[np.sum(edges_1),np.sum(edges_1 == 0)],
			[np.sum(edges_2),np.sum(edges_2 == 0)]])
		_,p = fisher_exact(table)
		ps[i,j] = p
		ps[j,i] = p

num_tests = P1.shape[0]*(P1.shape[0]-1)/2
alpha_corrected = 0.05/num_tests
edgewise_sig = np.sum(ps < alpha_corrected) - np.sum(n1) #all diagonals are 0
print("Number of significant edges from Fisher's exact with a=0.05, Bonferroni Correction: " + str(edgewise_sig))

vmin = np.log(alpha_corrected)

heatmap(ps,inner_hier_labels=lbls1,
	title ='Log-p for Edgewise Fisher Exact',
	transform='log',vmin=vmin,vmax=0)

plt.savefig('8_7_2019/fisher_edge.jpg')
#%%
#Blockwise test
print('Performing blockwise test...')
indices_1 = np.cumsum(n1)
num_blocks = indices_1.shape[0]
p_blocks = np.zeros((num_blocks,num_blocks))
lbls_block = np.arange(0,num_blocks)
for i in np.arange(num_blocks):
	if i==0:
		start_i = 0
	else:
		start_i = indices_1[i-1]
	end_i = indices_1[i]
	for j in np.arange(i,num_blocks):
		if j==0:
			start_j = 0
		else:
			start_j = indices_1[j-1]
		end_j = indices_1[j]
		
		block_1 = np.sum(P1[start_i:end_i,start_j:end_j,:],axis=2)
		block_2 = np.sum(P2[start_i:end_i,start_j:end_j,:],axis=2)

		#don't be redundant - only use upper triangle if on a diagonal block
		if i == j:
			idxs = np.triu_indices(n1[i],1)
			ones_1 = np.sum(block_1[idxs])
			zeros_1 = n1[i]*(n1[i]-1)*m/2 - ones_1
			ones_2 = np.sum(block_2[idxs])
			zeros_2 = n1[i]*(n1[i]-1)*m/2 - ones_2
		else:
			ones_1 = np.sum(block_1)
			zeros_1 = n1[i]**2*m - ones_1
			ones_2 = np.sum(block_2)
			zeros_2 = n1[j]**2*m - ones_2

		table = np.array([[ones_1,zeros_1],
			[ones_2,zeros_2]])
		
		_,p = fisher_exact(table)
		p_blocks[i,j] = p
		p_blocks[j,i] = p

num_tests = num_blocks*(num_blocks+1)/2
alpha_corrected = 0.05/num_tests
blockwise_sig = np.sum(p_blocks < alpha_corrected)
print("Number of significant blocks from Fisher's exact with a=0.05, Bonferroni Correction: " + str(blockwise_sig))

vmin = np.log(alpha_corrected)

heatmap(p_blocks,inner_hier_labels=lbls_block,
	title ='Log-p for Blockwise Fisher Exact',
	transform='log',vmin=vmin,vmax=0)

plt.savefig('8_7_2019/fisher_block.jpg')

#%%
#Magic
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.mgc import MGC

dcorr = DCorr()

indices_1 = np.cumsum(n1)
num_blocks = indices_1.shape[0]
p_blocks = np.zeros((num_blocks,num_blocks))
lbls_block = np.arange(0,num_blocks)
replication_factor = 1000

for i in np.arange(num_blocks):
	if i==0:
		start_i = 0
	else:
		start_i = indices_1[i-1]
	end_i = indices_1[i]
	for j in np.arange(i,num_blocks):
		if j==0:
			start_j = 0
		else:
			start_j = indices_1[j-1]
		end_j = indices_1[j]
		
		block_1 = np.sum(P1[start_i:end_i,start_j:end_j,:],axis=2)
		block_2 = np.sum(P2[start_i:end_i,start_j:end_j,:],axis=2)

		#don't be redundant - only use upper triangle if on a diagonal block
		if i == j:
			idxs = np.triu_indices(n1[i],1)
			edges_1 = block_1[idxs].flatten()
			edges_2 = block_2[idxs].flatten()
		else:
			edges_1 = block_1.flatten()
			edges_2 = block_2.flatten()

		u,v = k_sample_transform(edges_1,edges_2)
		p,_ = dcorr.p_value(u,v, replication_factor=replication_factor)

		if p < 1/replication_factor:
			p = 1/replication_factor

		p_blocks[i,j] = p
		p_blocks[j,i] = p

num_tests = num_blocks*(num_blocks+1)/2
alpha_corrected = 0.05/num_tests
blockwise_sig = np.sum(p_blocks < alpha_corrected)
print("Number of significant blocks from MGC with a=0.05, Bonferroni Correction: " + str(blockwise_sig))

vmin = np.log(alpha_corrected)

heatmap(p_blocks,inner_hier_labels=lbls_block,
	title ='Log-p for Blockwise MGC',
	transform='log',vmin=vmin,vmax=0)

plt.savefig('8_7_2019/mgc_block.jpg')
#%%
