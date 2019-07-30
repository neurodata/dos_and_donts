#%%
import numpy as np
from math import sqrt
from scipy.stats import pearsonr
from graspy.plot import heatmap
from sklearn.covariance import GraphicalLassoCV
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
np.random.seed(0)
#%%
#Generate x data
n=3
m=100
t=1000
#%%
#Null distribution of what we are doing
n_iter = 1000

null_distributions = np.zeros((4,n_iter))

for i in range(n_iter):
    data_test = np.random.normal(0,1,(2,t,m))
    node_0 = np.expand_dims(np.sum(data_test,axis=0),axis=0)
    data_test = np.append(node_0,data_test,axis=0)
    features = np.zeros((4,m))
    for j in range(m):
        cors = np.corrcoef(data_test[:,:,j])
        features[0,j] = cors[0,1] + np.random.normal(0,0.01)
        features[1,j] = cors[0,1]
        features[2,j] = cors[1,2]
        features[3,j] = cors[0,2]
    cors2 = np.corrcoef(features)
    invcors2 = np.linalg.inv(cors2)

    null_distributions[0,i] = cors2[0,1]
    null_distributions[1,i] = cors2[0,2]
    null_distributions[2,i] = invcors2[0,1]
    null_distributions[3,i] = invcors2[0,2]

fig, axs = plt.subplots(2,2)
axs[0,0].hist(null_distributions[0,:])
axs[0,1].hist(null_distributions[1,:])
axs[1,0].hist(null_distributions[2,:])
axs[1,1].hist(null_distributions[3,:])


#%%
