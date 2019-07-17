#%%
import matplotlib.pyplot as plt
import numpy as np

#Magic
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.mgc import MGC
from scipy.stats import fisher_exact

import matplotlib.pyplot as plt

import time
import progressbar



num_tests = 1000
num_vars = 100
p_vals = np.zeros((num_tests,1))

for i in progressbar.progressbar(range(num_tests)):

    d1 = np.random.binomial(100,0.2,(num_vars,1))
    d2 = np.random.binomial(100,0.2,(num_vars,1))

    u,v = k_sample_transform(d1,d2)
    dcorr = DCorr(which_test='unbiased')
    p,_ = dcorr.p_value(u,v)

    p_vals[i] = p


weights = np.ones_like(p_vals)/float(len(p_vals))
plt.hist(p_vals,bins=np.arange(0,1,0.01), weights=weights)
plt.xlabel('Dcorr P-vals')
plt.ylabel('Fraction of Occurrence')
plt.title('DCorr P-values for 2 sample test of iid Binomials (n=100,p=0.2,m=2500)')
#plt.savefig('nullps_2500.jpg')

#%%
'''
num_tests = 1
power_max = 13
num_varss = np.logspace(1,power_max,base=2,dtype=int)
p_vals = np.zeros((len(num_varss),1))
idx = 0

for num_vars in progressbar.progressbar(num_varss):
    for i in range(num_tests):
        d1 = np.random.binomial(100,0.5,(num_vars,1))
        d2 = np.random.binomial(100,0.51,(num_vars,1))

        u,v = k_sample_transform(d1,d2)
        dcorr = DCorr()
        p,_ = dcorr.p_value(u,v)

        p_vals[idx] = p
        idx = idx+1


plt.plot(num_varss,p_vals)
plt.xlabel('m')
plt.ylabel('P-val')
plt.title('DCorr P-values for 2-sample test of Binomials (n=100,p=0.5,0.51,m=m)')
plt.savefig('convergence_5000.jpg')
'''
#%%
