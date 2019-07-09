#%%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.dcorr import DCorr
import pandas as pd

from graspy.embed import OmnibusEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import sample_edges, sbm
from graspy.utils import cartprod
from src.utils import n_to_labels

sns.set_context("talk")
plt.style.use("seaborn-white")


#%% [markdown]
"""
pop 1 is a DC-SBM, 2 block, affinity the difference between it and pop 2 is that for
pop 1, the promiscuity parameter is 0.5 for all vertices and for pop 2, vertex 1 has a
different promiscuity parameter

similar analysis as before, but this time, we compare doing things "edge-wise" to doing
things "node wise" that is, we jointly embed (using mase or omni) and test whether each
vertex is different.

plots: 
heatmap of edge-wise p-values on log scale 
heatmap of node-wise p-values on a log scale

to get node-wise p-values, we embed each graph into 2D. 
hen we run 2-way mgc on each vertex to get a p-value. that gives us a p-value vector of
length n now, take the outer product of that vector with itself, to get an n x n matrix
and take the square root of that matrix. essentially, this should look like the
edge-wise p-value map "smoothed" by vertices.

lesson 7: model the nodes, not the edges
"""
#%% try redoing dcsbm
def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    # block_map = cartprod(inverse[0], inverse[1]).T
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


def dcsbm(vertex_assignments, block_p, degree_corrections):
    n_verts = len(vertex_assignments)
    p_mat = _block_to_full(block_p, vertex_assignments, (n_verts, n_verts))
    p_mat = p_mat * np.outer(degree_corrections, degree_corrections)
    return sample_edges(p_mat)


#%% Simulation setting: 2 populations of 2-block DCSBMs
# 8888 works when diff == 0 with randomized svd
# 8885 gives the flip... with randomized svd
np.random.seed(8889)
block_p = np.array([[0.25, 0.05], [0.05, 0.15]])
verts_per_block = 1000
n_verts = 2 * verts_per_block
n = 2 * [verts_per_block]
node_labels = n_to_labels(n).astype(int)
n_graphs = 20
diff = 0

vertex_assignments = np.zeros(n_verts, dtype=int)
vertex_assignments[verts_per_block:] = 1
degree_corrections = np.ones_like(vertex_assignments)


degree_corrections = np.ones(n_verts)
degree_corrections[0] += diff
degree_corrections[1:verts_per_block] -= diff / (verts_per_block - 1)

graphs_pop1 = []
for i in range(n_graphs):
    graphs_pop1.append(dcsbm(node_labels, block_p, degree_corrections))
# heatmap(graphs_pop1[0], inner_hier_labels=node_labels, cbar=False)

# modify the dcs for the next population
degree_corrections[0] += diff
degree_corrections[1:verts_per_block] -= diff / (verts_per_block - 1)

graphs_pop2 = []
for i in range(n_graphs):
    graphs_pop2.append(dcsbm(node_labels, block_p, degree_corrections))
# heatmap(graphs_pop2[0], inner_hier_labels=node_labels, cbar=False)

#%% Node-wise, embed and plot to see the 1 different node
n_components = 2

# mase = MultipleASE(n_components=n_components)
omni = OmnibusEmbed(n_components=n_components, algorithm="truncated")
pop1_latent = omni.fit_transform(graphs_pop1)
pop2_latent = omni.fit_transform(graphs_pop2)
labels1 = verts_per_block * ["Pop1 Block1"] + verts_per_block * ["Pop1 Block2"]
labels1 = np.tile(labels1, n_graphs)
labels2 = verts_per_block * ["Pop2 Block1"] + verts_per_block * ["Pop2 Block2"]
labels2 = np.tile(labels2, n_graphs)
labels = np.concatenate((labels1, labels2), axis=0)
plot_pop1_latent = pop1_latent.reshape((n_graphs * n_verts, n_components))
plot_pop2_latent = pop2_latent.reshape((n_graphs * n_verts, n_components))
plot_latents = np.concatenate((plot_pop1_latent, plot_pop2_latent), axis=0)
pairplot(plot_latents, labels=labels, alpha=0.3, height=4)

warnings.filterwarnings("ignore", category=UserWarning)

node_p_vals = []
node_metas = []
test = DCorr()
replication_factor = 100000

#%%


def node_wise_2_sample(node_ind):
    node_latent_pop1 = np.squeeze(pop1_latent[:, node_ind, :])
    node_latent_pop2 = np.squeeze(pop2_latent[:, node_ind, :])
    u, v = k_sample_transform(
        node_latent_pop1, node_latent_pop2, is_y_categorical=False
    )
    p_val, meta = test.p_value(u, v, replication_factor)
    if p_val < 1 / replication_factor:
        p_val = 1 / replication_factor
    return p_val


for node_ind in range(3):
    title = f"p-value: {node_wise_2_sample(node_ind):.3e}"
    node_latent_pop1 = pop1_latent[:, node_ind, :]  # all graphs, one node, all dims
    node_latent_pop2 = pop2_latent[:, node_ind, :]
    node_latent = np.concatenate((node_latent_pop1, node_latent_pop2), axis=0)
    pop_indicator = np.array(n_graphs * [0] + n_graphs * [1])
    pairplot(node_latent, labels=pop_indicator, title=title, height=4)

#%%

node_p_vals = Parallel(n_jobs=-2)(
    delayed(node_wise_2_sample)(i) for i in range(n_verts)
)
plot_data = pd.DataFrame(columns=["p value", "node index", "perturbed"])
plot_data["p value"] = node_p_vals
plot_data["node index"] = list(range(n_verts))
indicator = np.zeros(n_verts, dtype=bool)
indicator[0] = True
plot_data["perturbed"] = indicator
bonfer_thresh = 0.05 / n_verts
g = sns.scatterplot(data=plot_data, x="node index", y="p value", s=40)
plt.yscale("log")
plt.ylim([1e-6, 1])
plt.axhline(bonfer_thresh, c="r")
plt.savefig("exp7.pdf", format="pdf", facecolor="w")

#%%
