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
n=10
m=100
t=1000

data = np.random.normal(0,1,(n-1,t,m))
n0 = np.expand_dims(data[0,:,:] + data[1,:,:],axis=0)
data = np.concatenate((n0,data),axis=0)

#%%
#Create connectomes and generate y from connection 0-1
connectomes = []
y = []


for subject in range(m):
    connectome = np.corrcoef(data[:,:,subject])
    connectomes.append(connectome)
    y.append(connectome[0,1] + np.random.normal(0,0.01))

connectomes = np.stack(connectomes,axis=2)
y = np.array(y)

mean_connectome = np.mean(connectomes,axis=2)
heatmap(mean_connectome,title='Mean Connectome')

#%%
#Create feature vector which contains y and all connections
features = []
features.append(y)

for i in range(0,n):
    for j in range(i+1,n):
        edges = connectomes[i,j,:]
        features.append(edges)

features = np.stack(features,axis=1)


#%%
#All************************************graphical lasso
inds = np.triu_indices(n=n,k=1)

model = GraphicalLassoCV()
model.fit(features)
cov_ = model.covariance_
diags = np.power(np.diagonal(cov_),-1/2)
scale = np.diag(diags)
feature_corr_ = scale @ cov_ @ scale

edge_corrs = np.zeros((n,n))
edge_corrs[inds] = feature_corr_[0,1:]
edge_corrs = edge_corrs + edge_corrs.T
heatmap(edge_corrs,title='Correlations of Edges with Y')


print('Correlation of Y and Edge 0-1: %0.2f' % edge_corrs[0,1])
print('Correlation of Y and Edge 1-2: %0.2f' % edge_corrs[1,2])
#%%
prec_ = model.precision_
diags = np.power(np.diagonal(cov_),1/2)
scale = np.diag(diags)
feature_invcorr_ = scale @ prec_ @ scale

heatmap(feature_corr_ @ feature_invcorr_, title='Correlation * Inverse Correlation')

#%%

edge_invcorrs = np.zeros((n,n))
edge_invcorrs[inds] = feature_invcorr_[0,1:]
edge_invcorrs = edge_invcorrs + edge_invcorrs.T
heatmap(edge_invcorrs,title='Inverse correlations of Edges with Y')

print('InvCorrelation of Y and Edge 0-1: %0.2f' % edge_invcorrs[0,1])
print('InvCorrelation of Y and Edge 1-2: %0.2f' % edge_invcorrs[1,2])
#http://www.tulane.edu/~PsycStat/dunlap/Psyc613/RI2.html


#%%
#All**************************normal
feature_corr_ = np.corrcoef(features.T)
edge_corrs = np.zeros((n,n))
edge_corrs[inds] = feature_corr_[0,1:]
edge_corrs = edge_corrs + edge_corrs.T
heatmap(edge_corrs,title='Correlations of Edges with Y')

print('Correlation of Y and Edge 0-1: %0.2f' % edge_corrs[0,1])
print('Correlation of Y and Edge 1-2: %0.2f' % edge_corrs[1,2])
#%%
feature_invcorr_ = np.linalg.inv(feature_corr_)

heatmap(feature_corr_ @ feature_invcorr_, title='Correlation * Inverse Correlation')

#%%

edge_invcorrs = np.zeros((n,n))
edge_invcorrs[inds] = feature_invcorr_[0,1:]
edge_invcorrs = edge_invcorrs + edge_invcorrs.T
heatmap(edge_invcorrs,inner_hier_labels=[0,1,2],title='Inverse correlations of Edges with Y')

print('InvCorrelation of Y and Edge 0-1: %0.2f' % edge_invcorrs[0,1])
print('InvCorrelation of Y and Edge 1-2: %0.2f' % edge_invcorrs[1,2])


#%%
#Only relevant************************graphical lasso
features2 = features[:,[0,1,3]]
model = GraphicalLassoCV()
model.fit(features2)
cov_ = model.covariance_
diags = np.power(np.diagonal(cov_),-1/2)
scale = np.diag(diags)
feature_corr_ = scale @ cov_ @ scale

edge_corrs = np.zeros((n,n))
edge_corrs[0,1] = feature_corr_[0,1]
edge_corrs[1,2] = feature_corr_[0,2]
edge_corrs = edge_corrs + edge_corrs.T
heatmap(edge_corrs,inner_hier_labels=[0,1,2],title='Correlations of Edges with Y')


print('Correlation of Y and Edge 0-1: %0.2f' % edge_corrs[0,1])
print('Correlation of Y and Edge 1-2: %0.2f' % edge_corrs[1,2])
#%%
prec_ = model.precision_
diags = np.power(np.diagonal(cov_),1/2)
scale = np.diag(diags)
feature_invcorr_ = scale @ prec_ @ scale

heatmap(feature_corr_ @ feature_invcorr_, title='Correlation * Inverse Correlation')
#%%

edge_invcorrs = np.zeros((n,n))
edge_invcorrs[0,1] = feature_invcorr_[0,1]
edge_invcorrs[1,2] = feature_invcorr_[0,2]
edge_invcorrs = edge_invcorrs + edge_invcorrs.T
heatmap(edge_invcorrs,inner_hier_labels=[0,1,2],title='Inverse correlations of Edges with Y')

print('InvCorrelation of Y and Edge 0-1: %0.2f' % edge_invcorrs[0,1])
print('InvCorrelation of Y and Edge 1-2: %0.2f' % edge_invcorrs[1,2])
