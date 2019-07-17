#%%
import numpy as np
from math import sqrt
from scipy.stats import pearsonr
np.random.seed(0)
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
print('n0 vs n1 correlation: %.2f Pearson test p-value: %.2f' %(cors[0,1],p01))
_,p02 = pearsonr(data_nxtm[0,:],data_nxtm[2,:])
print('n0 vs n2 correlation: %.2f Pearson test p-value: %.2f' %(cors[0,2],p02))
_,p12 = pearsonr(data_nxtm[1,:],data_nxtm[2,:])
print('n1 vs n2 correlation: %.2f Pearson test p-value: %.2f' %(cors[1,2],p12))
#%%
#Generate y data
y = np.zeros(m)
cors_02 = np.zeros(m)
cors_01 = np.zeros(m)

for subject in range(m):
    noise = np.random.normal(0,0.03)
    corrs = np.corrcoef(data[:,:,subject])
    y[subject] = noise + corrs[0,1]
    cors_01[subject] = corrs[0,1]
    cors_02[subject] = corrs[0,2]

#%%
#Measure correlations (of correlations) with y

# n0 and n2 are independent but both compose n1.
# n0 correlated with n2 => n0 highly correlated with n1 => high y
underlying_cor = np.corrcoef(y,cors_01)[0,1]
_,p01 = pearsonr(y,cors_01)
misleading_cor = np.corrcoef(y,cors_02)[0,1]
_,p02 = pearsonr(y,cors_02)

print('y vs Correlation(n0,n1) correlation: %.2f Pearson test p-value: %.2f' % (underlying_cor, p01))
print('y vs Correlation(n0,n2) correlation: %.2f Pearson test p-value: %.2f' % (misleading_cor,p02))

#%%
#Measure correlations (of inverse correlations) with y
invcors_02 = np.zeros(m)
invcors_01 = np.zeros(m)

for subject in range(m):
    corrs = np.corrcoef(data[:,:,subject])
    invcorrs = np.linalg.inv(corrs)
    invcors_01[subject] = invcorrs[0,1]
    invcors_02[subject] = invcorrs[0,2]

underlying_cor = np.corrcoef(y,invcors_01)[0,1]
_,p01 = pearsonr(y,invcors_01)
misleading_cor = np.corrcoef(y,invcors_02)[0,1]
_,p02 = pearsonr(y,invcors_02)

print('y vs invCorrelation(n0,n1) correlation: %.2f Pearson test p-value: %.2f' % (underlying_cor, p01))
print('y vs invCorrelation(n0,n2) correlation: %.2f Pearson test p-value: %.2f' %(misleading_cor,p02))
#http://www.tulane.edu/~PsycStat/dunlap/Psyc613/RI2.html

#%%
