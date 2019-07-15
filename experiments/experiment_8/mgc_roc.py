#%%
import matplotlib.pyplot as plt
import numpy as np

#Magic
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.mgc import MGC

import matplotlib.pyplot as plt

dcorr = DCorr()

alphas = np.arange(0.05,0.95,0.1)
fprs = np.zeros(alphas.size)

num_tests = 10
replication_factor = 1000

for j,alpha in enumerate(alphas):
    print(alpha)
    positive_count = 0
    for i in range(num_tests):
        d1 = np.random.binomial(,2,(50,50,100))
        d1 = np.sum(d1,axis=2).flatten()

        d2 = np.random.randint(0,2,(50,50,100))
        d2 = np.sum(d2,axis=2).flatten()

        u,v = k_sample_transform(d1,d2)
        p,_ = dcorr.p_value(u,v,replication_factor=replication_factor)

        if p < 1/replication_factor:
            p = 1/replication_factor
        
        if p < alpha:
            count = count + 1
    fprs[j] = float(count)/num_tests


        
plt.plot(alphas, fprs)

#%%
