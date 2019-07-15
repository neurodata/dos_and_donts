#%%
import numpy as np
from scipy.special import comb 
import matplotlib.pyplot as plt

w=20
n_std = 4



p1 = 0.5
p2 = 0.5
n1 = 100
n2 = 100

p2s = np.arange(p1,p1+0.2,0.05)
prob_sigs = np.zeros(len(p2s))
for i,p2 in enumerate(p2s):

    std_1 = np.sqrt(n1*p1*(1-p1))
    std_2 = np.sqrt(n2*p2*(1-p2))

    range_1 = np.arange(int(p1*n1)-n_std*std_1, int(p1*n1)+n_std*std_1+1)
    range_1 = np.expand_dims(range_1,axis=1)

    range_2 = np.arange(int(p2*n2)-n_std*std_2, int(p2*n2)+n_std*std_2+1)
    range_2 = np.expand_dims(range_2,axis=1)

    choice_1 = comb(n1,range_1)
    term_1 = np.multiply(np.power(p1,range_1),np.power(1-p1,n1-range_1))
    prob_1 = np.multiply(choice_1,term_1)

    choice_2 = comb(n2,range_2)
    term_2 = np.multiply(np.power(p2,range_2),np.power(1-p2,n2-range_2))
    prob_2 = np.multiply(choice_2,term_2)


    prob_joint = np.matmul(prob_1,prob_2.T)
    print("Total probability in this region: "
        + str(np.sum(prob_joint)))


    p_hat_1 = np.repeat(range_1/n1,range_2.size,axis=1)
    p_hat_2 = np.repeat(range_2.T/n1,range_1.size,axis=0)

    p_hat = (n1*p_hat_1+n2*p_hat_2)/(n1+n2)
    sigma_pooled = np.sqrt((1/n1+1/n2)*np.multiply(p_hat,1-p_hat))
    z = np.divide(np.absolute(p_hat_1-p_hat_2)-0.5*(1/n1+1/n2),sigma_pooled)
    sig = z>1.645
    prob_sig = np.sum(np.multiply(sig,prob_joint))
    prob_sigs[i] = prob_sig
    print(prob_sig)

plt.plot(p2s, prob_sigs)
plt.ylabel('Probability')
plt.xlabel('p2')
plt.title('Probability that Chi-sq test is significant (p1=0.5, n1=n2=100)')

plt.savefig('chisq_sig.jpg')

#%%
