#%%
import numpy as np
from graspy.plot import heatmap, gridplot
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
mat1 = np.array([[0.2, 0.05], [0.1, 0.15]])
mat2 = np.array([[0.2, 0.05], [0.1, 0.15]])
mats = [mat1, mat2]
labels = [0, 1]
heatmap(mat1, inner_hier_labels=labels, sort_nodes=False, vmax=1)
#%%
from graspy.simulations import sbm

mat1 = np.array([[1, 1], [1, 1]])
n_per_block = 4
labels = n_per_block * [0] + n_per_block * [1]
graphs = [sbm([n_per_block, n_per_block], mat1, directed=True) for _ in range(2)]
gridplot(graphs, inner_hier_labels=labels)


#%%
heatmap(graphs[0], inner_hier_labels=labels)


#%%
