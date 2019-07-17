#%%
import numpy as np
from math import sqrt
from scipy.stats import pearsonr

n1 = np.random.normal(0,1,(100,10))
n3 = np.random.normal(0,1,(100,10))
n2 = (n1+n3)*1/sqrt(2)

#nxtxm
data = np.stack((n1,n2,n3),axis=0)
data_nxtm = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
cov = np.cov(data_nxtm)

_,p01 = pearsonr(data_nxtm[0,:],data_nxtm[1,:])
print('Pearson test p-value n0 vs n1: ' + str(p01))
_,p02 = pearsonr(data_nxtm[0,:],data_nxtm[2,:])
print('Pearson test p-value n0 vs n2: ' + str(p02))
_,p12 = pearsonr(data_nxtm[1,:],data_nxtm[2,:])
print('Pearson test p-value n1 vs n2: ' + str(p12))
#%%
