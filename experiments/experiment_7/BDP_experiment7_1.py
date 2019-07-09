#%%
import warnings

import mgcpy
import numpy as np
from joblib import Parallel, delayed
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC

from graspy.embed import MultipleASE, OmnibusEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import sample_edges, sbm
from graspy.utils import cartprod
from src.utils import n_to_labels

plot = False
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
block_p = np.array([[0.15, 0.05], [0.05, 0.15]])
verts_per_block = 100
n_verts = 2 * verts_per_block
n = 2 * [verts_per_block]
node_labels = n_to_labels(n).astype(int)
n_graphs = 200
diff = 0.25

vertex_assignments = np.zeros(n_verts, dtype=int)
vertex_assignments[verts_per_block:] = 1
degree_corrections = np.ones_like(vertex_assignments)


degree_corrections = np.ones(n_verts)
graphs_pop1 = []
for i in range(n_graphs):
    graphs_pop1.append(dcsbm(node_labels, block_p, degree_corrections))
# heatmap(graphs_pop1[0], inner_hier_labels=node_labels, cbar=False)

# modify the dcs for the next population
degree_corrections[0] += diff
degree_corrections[1:verts_per_block] -= diff / (verts_per_block - 1)
print(np.mean(degree_corrections[:verts_per_block]))

graphs_pop2 = []
for i in range(n_graphs):
    graphs_pop2.append(dcsbm(node_labels, block_p, degree_corrections))
# heatmap(graphs_pop2[0], inner_hier_labels=node_labels, cbar=False)
# #%% old sim
# dc_pop1 = np.array(sum(n) * [1 / verts_per_block])
# dc_pop2 = np.array(sum(n) * [1 / verts_per_block])
# dc_pop2[0] = dc_pop2[0] + diff
# dc_pop2[1:verts_per_block] = (1 - dc_pop2[0]) / (verts_per_block - 1)

# graphs_pop1 = []
# for i in range(n_graphs):
#     graphs_pop1.append(sbm(n, block_p, dc=dc_pop1))
# heatmap(graphs_pop1[0], inner_hier_labels=pop_labels, cbar=False)

# graphs_pop2 = []
# for i in range(n_graphs):
#     graphs_pop2.append(sbm(n, block_p, dc=dc_pop2))
# heatmap(graphs_pop2[0], inner_hier_labels=pop_labels, cbar=False)
#%%
pop2graphs = np.array(graphs_pop2)
p2 = np.mean(pop2graphs[:, :verts_per_block, :verts_per_block])
p2

pop1graphs = np.array(graphs_pop1)
p1 = np.mean(pop1graphs[:, :verts_per_block, :verts_per_block])
p1
#%% Node-wise, embed and plot to see the 1 different node
plot = True
n_components = 2

# mase = MultipleASE(n_components=n_components)
omni = OmnibusEmbed(n_components=n_components)
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
if plot:
    pairplot(plot_latents, labels=labels, alpha=0.3)

warnings.filterwarnings("ignore", category=UserWarning)

node_p_vals = []
node_metas = []
test = DCorr()
replication_factor = 10000

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
    title = str(node_wise_2_sample(node_ind))
    node_latent_pop1 = pop1_latent[:, node_ind, :]  # all graphs, one node, all dims
    #     print(node_latent_pop1.shape)
    node_latent_pop2 = pop2_latent[:, node_ind, :]
    node_latent = np.concatenate((node_latent_pop1, node_latent_pop2), axis=0)
    #     print(node_latent.shape)
    pop_indicator = np.array(n_graphs * [0] + n_graphs * [1])
    #     print(pop_indicator)
    pairplot(node_latent, labels=pop_indicator, title=title)
#     pop_indicator = pop_indicator[:, np.newaxis]
#     u, v = k_sample_transform(pop1_latent, pop2_latent, is_y_categorical=False)
#     # mgc = MGC()
#     p_val, meta = test.p_value(u, v, 1000)
#     print(p_val)
#     node_p_vals.append(p_val)
#     node_metas.append(meta)
# node_p_vals = np.array(node_p_vals)


# def node_wise_2_sample(node_ind):
#     node_latent_pop1 = pop1_latent[:, node_ind, :]  # all graphs, one node, all dims
#     node_latent_pop2 = pop2_latent[:, node_ind, :]
#     node_latent = np.concatenate((node_latent_pop1, node_latent_pop2), axis=0)
#     pop_indicator = np.array(n_graphs * [0] + n_graphs * [1])
#     pop_indicator = pop_indicator[:, np.newaxis]
#     u, v = k_sample_transform(node_latent, pop_indicator, is_y_categorical=True)
#     p_val, meta = test.p_value(u, v, replication_factor)
#     if p_val == 0:
#         p_val = 1 / replication_factor
#     return p_val


#%%
node_p_vals = Parallel(n_jobs=-2)(
    delayed(node_wise_2_sample)(i) for i in range(n_verts)
)

#%%
p_val_mat = np.sqrt(np.outer(node_p_vals, node_p_vals))
bonfer_thresh = 0.05 / n_verts
# heatmap(p_val_mat, transform="log", vmax=bonfer_thresh, vmin=0)
#%%
sns.lineplot(x=list(range(n_verts)), y=node_p_vals)
plt.yscale("log")
plt.axhline(bonfer_thresh)
# #%%
# log_p = np.zeros(P1.shape[:2])
# for i in range(P1.shape[0]):
#     for j in range(i + 1, P1.shape[1]):
#         edges_1 = P1[i, j, :]
#         edges_2 = P2[i, j, :]
#         table = np.array(
#             [
#                 [np.sum(edges_1), np.sum(edges_1 == 0)],
#                 [np.sum(edges_2), np.sum(edges_2 == 0)],
#             ]
#         )
#         _, p = fisher_exact(table)
#         log_p[i, j] = np.log(p)
#         log_p[j, i] = np.log(p)

# num_tests = P1.shape[0] * (P1.shape[0] - 1) / 2
# edgewise_sig = np.sum(log_p < np.log(0.05 / num_tests))
# print(
#     "Number of significant edges from Fisher's exact with a=0.05, Bonferroni Correction: "
#     + str(edgewise_sig)
# )

# heatmap(log_p, inner_hier_labels=lbls1, title="Log-p for Edgewise Fisher Exact")


#%%
