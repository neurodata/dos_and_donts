#%%
import numpy as np 
import matplotlib.pyplot as plt

alpha = 0.05

n = np.arange(50.*99/2,50.*99/2+2)
beta_1 = np.divide(alpha,n)
beta_2 = 1 - np.power(0.95,np.reciprocal(n))


plt.plot(n,beta_1,label='Bonferroni')
plt.plot(n,beta_2,label='Exact')

plt.legend()

#%%
