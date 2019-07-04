#%%
from graspy.simulations import sbm
from graspy.plot import heatmap
import numpy as np
from src.utils import n_to_labels
from graspy.embed import OmnibusEmbed

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
#%%
block_p = np.array([[0.15, 0.05], [0.05, 0.15]])
verts_per_block = 100
n = 2 * [verts_per_block]
pop_labels = n_to_labels(n)
n_graphs = 10

diff = 0.0005
dc_pop1 = np.array(sum(n) * [1 / verts_per_block])
dc_pop2 = np.array(sum(n) * [1 / verts_per_block])
dc_pop2[0] = 0.5 + diff
dc_pop2[1:verts_per_block] = (0.5 - diff) / (verts_per_block - 1)

graphs_pop1 = []
for i in range(n_graphs):
    graphs_pop1.append(sbm(n, block_p, dc=dc_pop1))
heatmap(graphs_pop1[0], inner_hier_labels=pop_labels, cbar=False)

graphs_pop2 = []
for i in range(n_graphs):
    graphs_pop2.append(sbm(n, block_p, dc=dc_pop2))
heatmap(graphs_pop2[0], inner_hier_labels=pop_labels, cbar=False)


#%% Node-wise
omni = OmnibusEmbed(n_components=2)
pop1_latent = omni.fit_transform(graphs_pop1)

#%%
log_p = np.zeros(P1.shape[:2])
for i in range(P1.shape[0]):
    for j in range(i + 1, P1.shape[1]):
        edges_1 = P1[i, j, :]
        edges_2 = P2[i, j, :]
        table = np.array(
            [
                [np.sum(edges_1), np.sum(edges_1 == 0)],
                [np.sum(edges_2), np.sum(edges_2 == 0)],
            ]
        )
        _, p = fisher_exact(table)
        log_p[i, j] = np.log(p)
        log_p[j, i] = np.log(p)

num_tests = P1.shape[0] * (P1.shape[0] - 1) / 2
edgewise_sig = np.sum(log_p < np.log(0.05 / num_tests))
print(
    "Number of significant edges from Fisher's exact with a=0.05, Bonferroni Correction: "
    + str(edgewise_sig)
)

heatmap(log_p, inner_hier_labels=lbls1, title="Log-p for Edgewise Fisher Exact")
