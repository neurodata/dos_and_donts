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

from os.path import basename
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.dcorr import DCorr
from scipy.stats import fisher_exact

from graspy.embed import OmnibusEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import p_from_latent, sample_edges
from graspy.utils import cartprod
from src.utils import n_to_labels

folderpath = Path(__file__.replace(basename(__file__), ""))

sns.set_context("talk", font_scale=1.5)
plt.style.use("seaborn-white")
sns.set_palette("deep")
np.random.seed(88888)
savefig = True
fmt = "pdf"
savefig_kws = dict(format=fmt, facecolor="w")
font_scale = 1.5

#%% functions


def block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


def dcsbm(vertex_assignments, block_p, degree_corrections, return_p_mat=False):
    n_verts = len(vertex_assignments)
    p_mat = block_to_full(block_p, vertex_assignments, (n_verts, n_verts))
    p_mat = p_mat * np.outer(degree_corrections, degree_corrections)
    if return_p_mat:
        return p_mat
    else:
        return sample_edges(p_mat, directed=False, loops=True)


def sample_graph(latent):
    p = p_from_latent(latent, rescale=False, loops=False)
    return sample_edges(p, directed=False, loops=False)


def compute_t_stat(sample1, sample2):
    test = DCorr()
    u, v = k_sample_transform(sample1, sample2, is_y_categorical=False)
    return test.test_statistic(u, v)[0]


def node_wise_2_sample(latent, node_ind):
    """Get dcorr t stat for a single node"""
    node_latent_pop1 = np.squeeze(latent[:n_graphs, node_ind, :])
    node_latent_pop2 = np.squeeze(latent[n_graphs:, node_ind, :])
    t_stat = compute_t_stat(node_latent_pop1, node_latent_pop2)
    return t_stat


def compute_pop_t_stats(pop_latent):
    """assumes the first half are from one pop"""
    n_verts = pop_latent.shape[1]
    t_stats = np.zeros(n_verts)
    for node_ind in range(n_verts):
        t_stat = node_wise_2_sample(pop_latent, node_ind)
        t_stats[node_ind] = t_stat
    return t_stats


def bootstrap_population(latent, n_graphs, seed):
    np.random.seed(seed)
    bootstrapped_graphs = []
    for g in range(n_graphs):
        graph = sample_graph(latent)
        bootstrapped_graphs.append(graph)

    omni = OmnibusEmbed(n_components=2)
    bootstrapped_latent = omni.fit_transform(bootstrapped_graphs)
    bootstrap_t_stats = compute_pop_t_stats(bootstrapped_latent)
    return bootstrap_t_stats


def run_edgewise_2_sample(pop1, pop2, method):
    n_verts = pop1[0, :, :].shape[0]
    out_mat = np.zeros((n_verts, n_verts))
    for i in range(n_verts):
        for j in range(n_verts):
            sample1 = pop1[:, i, j]
            sample2 = pop2[:, i, j]
            out = method(sample1, sample2)
            out_mat[i, j] = out
    return out_mat


def run_fishers(sample1, sample2):
    table = np.zeros((2, 2))
    table[0, 0] = np.sum(sample1 == 1)
    table[0, 1] = np.sum(sample1 == 0)
    table[1, 0] = np.sum(sample2 == 1)
    table[1, 1] = np.sum(sample2 == 0)
    _, p = fisher_exact(table)
    return p


#%% Simulation setting: 2 populations of 2-block DCSBMs

# Simulation parameters
block_p = np.array([[0.25, 0.05], [0.05, 0.15]])
verts_per_block = 100
n_verts = 2 * verts_per_block
n = 2 * [verts_per_block]
node_labels = n_to_labels(n).astype(int)
n_graphs = 10
diff = 1

print("Generating graph populations")
vertex_assignments = np.zeros(n_verts, dtype=int)
vertex_assignments[verts_per_block:] = 1
degree_corrections = np.ones(n_verts)

# Population 1
p_mat = dcsbm(node_labels, block_p, degree_corrections, return_p_mat=True)
heatmap(
    p_mat,
    inner_hier_labels=node_labels,
    cbar=True,
    vmin=0,
    vmax=1,
    title="Pop1 P matrix",
    font_scale=font_scale,
)
if savefig:
    plt.savefig(folderpath / f"pop1_p_mat.{fmt}", **savefig_kws)
graphs_pop1 = []
for i in range(n_graphs):
    graphs_pop1.append(dcsbm(node_labels, block_p, degree_corrections))
graphs_pop1 = np.array(graphs_pop1)
heatmap(
    graphs_pop1[0],
    inner_hier_labels=node_labels,
    cbar=False,
    title="Pop1 sample",
    font_scale=font_scale,
)
if savefig:
    plt.savefig(folderpath / f"pop1_sample.{fmt}", **savefig_kws)

# Population 2
degree_corrections[0] += diff
degree_corrections[1:verts_per_block] -= diff / (verts_per_block - 1)

p_mat = dcsbm(node_labels, block_p, degree_corrections, return_p_mat=True)
heatmap(
    p_mat,
    inner_hier_labels=node_labels,
    cbar=True,
    vmin=0,
    vmax=1,
    title="Pop2 P matrix",
    font_scale=font_scale,
)
if savefig:
    plt.savefig(folderpath / f"pop2_p_mat.{fmt}", **savefig_kws)
graphs_pop2 = []
for i in range(n_graphs):
    graphs_pop2.append(dcsbm(node_labels, block_p, degree_corrections))
graphs_pop2 = np.array(graphs_pop2)
heatmap(
    graphs_pop2[0],
    inner_hier_labels=node_labels,
    cbar=False,
    title="Pop2 sample",
    font_scale=font_scale,
)
if savefig:
    plt.savefig(folderpath / f"pop2_sample.{fmt}", **savefig_kws)

#%% Omni embedding of population
n_components = 2

print("Doing Omnibus Embedding")

omni = OmnibusEmbed(n_components=n_components, algorithm="randomized")
graphs = np.concatenate((graphs_pop1, graphs_pop2), axis=0)
pop_latent = omni.fit_transform(graphs)
labels1 = verts_per_block * ["Pop1 Block1"] + verts_per_block * ["Pop1 Block2"]
labels1 = np.tile(labels1, n_graphs)
labels2 = verts_per_block * ["Pop2 Block1"] + verts_per_block * ["Pop2 Block2"]
labels2 = np.tile(labels2, n_graphs)
labels = np.concatenate((labels1, labels2), axis=0)
plot_pop_latent = pop_latent.reshape((2 * n_graphs * n_verts, n_components))
pairplot(plot_pop_latent, labels=labels, alpha=0.5, height=4, title="Omnibus embedding")
if savefig:
    plt.savefig(folderpath / f"pop_pairplot.{fmt}", **savefig_kws)

#%% Bootstrapping

n_bootstraps = 10
print(f"Running {n_bootstraps} bootstraps")


avg_latent = np.mean(pop_latent, axis=0)


def bsp(seed):
    return bootstrap_population(avg_latent, n_graphs * 2, seed)


seeds = np.random.randint(1e8, size=n_bootstraps)
out = Parallel(n_jobs=-2, verbose=10)(delayed(bsp)(seed) for seed in seeds)
nulls = np.array(out).T

sample_t_stats = compute_pop_t_stats(pop_latent)
node_p_vals = np.zeros(n_verts)
for i, sample_t in enumerate(sample_t_stats):
    num_greater = len(np.where(sample_t < nulls[i, :])[0])
    p_val = num_greater / n_bootstraps
    if p_val < 1 / n_bootstraps:
        p_val = 1 / n_bootstraps
    node_p_vals[i] = p_val


#%% this plot would show individual node embeddings
# for node_ind in range(3):
#     title = f"p-value: {node_wise_2_sample(node_ind):.3e}"
#     node_latent_pop1 = pop_latent[:n_graphs, node_ind, :]
#     node_latent_pop2 = pop_latent[n_graphs:, node_ind, :]
#     node_latent = np.concatenate((node_latent_pop1, node_latent_pop2), axis=0)
#     pop_indicator = np.array(n_graphs * [0] + n_graphs * [1])
#     pairplot(node_latent, labels=pop_indicator, title=title, height=4)

#%% plot results of nodewise test

plot_data = pd.DataFrame(columns=["p value", "node index", "perturbed"])
plot_data["p value"] = node_p_vals
plot_data["node index"] = list(range(n_verts))
indicator = np.zeros(n_verts, dtype=bool)
indicator[0] = True
plot_data["perturbed"] = indicator
bonfer_thresh = 0.05 / n_verts

plt.style.use("seaborn-white")
plt.figure(figsize=(20, 10))
g = sns.scatterplot(data=plot_data, x="node index", y="p value", s=40, hue="perturbed")

plt.yscale("log")
plt.ylim([1e-8, 1])
color = sns.color_palette("Set1", desat=0.7)[0]
plt.axhline(bonfer_thresh, c=color, linestyle="--")
plt.title(f"Nodewise omnibus bootstrap, n={n_verts}, m={2*n_graphs}")

plot_data.to_csv(folderpath / "nodewise_p_vals.csv")
if savefig:
    plt.savefig(folderpath / f"nodewise_p_vals.{fmt}", **savefig_kws)

print("Saved, done with nodewise bootstrap")

#%% run fisher's exact on each edge
print("Running Fisher's exact")
edgewise_pval_mat = run_edgewise_2_sample(graphs_pop1, graphs_pop2, run_fishers)

num_tests = n_verts * (n_verts - 1) / 2
sig_level = 0.05 / num_tests

sig_level_log = np.log10(sig_level)
edgewise_plot = np.log10(edgewise_pval_mat)

heatmap(
    edgewise_plot,
    inner_hier_labels=node_labels,
    transform="log",
    cmap="RdBu",
    vmin=-10,
    vmax=0,
    center=0,
    title=f"Edgewise Fisher's, SL={sig_level_log:.2f}",
    font_scale=font_scale,
)
if savefig:
    plt.savefig(folderpath / f"edgewise_p_vals.{fmt}", **savefig_kws)

print("Done!")

