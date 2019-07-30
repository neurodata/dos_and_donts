#%%
import numpy as np
from math import sqrt
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
#np.random.seed(0)

#Generate x data
reps = 1000 #number of samples
m=100 #number of subjects
t=1000 #number of time points

c_01_y = np.zeros(reps) #correlations between cor(n1,)
ic_01_y = np.zeros(reps)
ic_02_y = np.zeros(reps)

for i in range(reps):
    n0 = np.random.normal(0,1,(t,m))
    n1 = np.random.normal(0,1,(t,m))
    noise = np.random.normal(0,1,(t,m))
    n2 = (n0+n1) + noise 

    #nxtxm
    data = np.stack((n0,n1,n2),axis=0)

    cors_02 = np.zeros(m)
    cors_01 = np.zeros(m)
    invcors_01 = np.zeros(m)
    invcors_02 = np.zeros(m)

    for subject in range(m):
        corrs = np.corrcoef(data[:,:,subject])
        invcorrs = np.linalg.inv(corrs)

        cors_02[subject] = corrs[0,2]
        cors_01[subject] = corrs[0,1]
        invcors_01[subject] = invcorrs[0,1]
        invcors_02[subject] = invcorrs[0,2]
    
    c_01_y[i] = np.corrcoef(cors_02,cors_01)[0,1]
    ic_02_y[i] = np.corrcoef(cors_02,invcors_02)[0,1]
    ic_01_y[i] = np.corrcoef(cors_02,invcors_01)[0,1]

c_01_y_runav = [np.sum(c_01_y[0:i])/i for i in range(1,len(c_01_y)+1)]
ic_02_y_runav = [np.sum(ic_02_y[0:i])/i for i in range(1,len(ic_02_y)+1)]
ic_01_y_runav = [np.sum(ic_01_y[0:i])/i for i in range(1,len(ic_01_y)+1)]


plt.plot(range(reps),c_01_y_runav,label='cor(01)')
plt.plot(range(reps),ic_02_y_runav,label='invcor(02)')
plt.plot(range(reps), ic_01_y_runav,label='invcor(01)')
plt.plot(np.arange(reps),0*np.arange(reps))
plt.xlabel('# samples')
plt.ylabel('Correlation coefficient')
plt.title('Correlations with cor(0,2)')
plt.legend()
plt.show()
#%%

