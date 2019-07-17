#%%
import numpy as np
from math import sqrt
from scipy.stats import pearsonr
#%%
#Generate x data
m=100
t=1000

n1 = np.random.normal(0,1,(t,m))
n3 = np.random.normal(0,1,(t,m))
n2 = (n1+n3)*1/sqrt(2)

#nxtxm
data = np.stack((n1,n2,n3),axis=0)
data_nxtm = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
cors = np.corrcoef(data_nxtm)

_,p01 = pearsonr(data_nxtm[0,:],data_nxtm[1,:])
print('n0 vs n1 correlation: ' + str(cors[0,1]) + ' Pearson test p-value: ' + str(p01))
_,p02 = pearsonr(data_nxtm[0,:],data_nxtm[2,:])
print('n0 vs n2 correlation: ' + str(cors[0,2]) + ' Pearson test p-value: ' + str(p02))
_,p12 = pearsonr(data_nxtm[1,:],data_nxtm[2,:])
print('n1 vs n2 correlation: ' + str(cors[1,2]) + ' Pearson test p-value: ' + str(p12))
#%%
#Generate y data
y = np.zeros(m)
cors_02 = np.zeros(m)

for subject in range(m):
    noise = 0#np.random.normal(0,1)
    corrs = np.corrcoef(data[:,:,subject],data[:,:,subject])
    y[subject] = noise + corrs[0,1]
    cors_02[subject] = corrs[0,2]

misleading_cov = np.corrcoef(y,cors_02)[0,1]
_,p = pearsonr(y,cors_02)
print('y vs Correlation(n0,n2) correlaiton: ' + str(misleading_cov) + ' ' + str(p))

#%%
