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
from tqdm import tqdm
from graspy.embed import OmnibusEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import p_from_latent, sample_edges
from graspy.utils import cartprod
from src.utils import n_to_labels

sns.set_context("talk", font_scale=1.5)
plt.style.use("seaborn-white")
sns.set_palette("deep")
np.random.seed(88888)
font_scale = 1.5

folderpath = Path(__file__.replace(basename(__file__), ""))
savepath = folderpath / "outputs"


def block_to_full(block_mat, inverse, shape):
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
    node_latent_pop1 = np.squeeze(latent[:n_graphs, node_ind, :])
    node_latent_pop2 = np.squeeze(latent[n_graphs:, node_ind, :])
    t_stat = compute_t_stat(node_latent_pop1, node_latent_pop2)
    return t_stat


def compute_pop_t_stats(pop_latent):
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


# Simulation setting: 2 populations of 2-block DCSBMs

block_p = np.array([[0.25, 0.05], [0.05, 0.15]])
n_graphs = 10
diff = 1
verts_per_block = 100
n_verts = 2 * verts_per_block
n = 2 * [verts_per_block]
node_labels = n_to_labels(n).astype(int)
temp = []
node1 = []
node_list = []

# test settings
sims = 200
n_bootstraps = 10000
verbose_parallel = 0
verbose = False
n_jobs = -2

for x in tqdm(range(sims)):
    if verbose:
        print(f"Running simulation {x+1}")

    if verbose:
        print("Generating graph populations")
    vertex_assignments = np.zeros(n_verts, dtype=int)
    vertex_assignments[verts_per_block:] = 1
    degree_corrections = np.ones(n_verts)

    # Population 1
    graphs_pop1 = []
    for i in range(n_graphs):
        graphs_pop1.append(dcsbm(vertex_assignments, block_p, degree_corrections))
    graphs_pop1 = np.array(graphs_pop1)

    # Population 2
    degree_corrections[0] += diff
    degree_corrections[1:verts_per_block] -= diff / (verts_per_block - 1)

    graphs_pop2 = []
    for i in range(n_graphs):
        graphs_pop2.append(dcsbm(node_labels, block_p, degree_corrections))
    graphs_pop2 = np.array(graphs_pop2)

    n_components = 2

    if verbose:
        print("Doing Omnibus Embedding")
    omni = OmnibusEmbed(n_components=n_components, algorithm="randomized")
    graphs = np.concatenate((graphs_pop1, graphs_pop2), axis=0)
    pop_latent = omni.fit_transform(graphs)

    # Bootstrapping
    if verbose:
        print(f"Running {n_bootstraps} bootstraps")

    avg_latent = np.mean(pop_latent, axis=0)

    def bsp(seed):
        return bootstrap_population(avg_latent, n_graphs * 2, seed)

    seeds = np.random.randint(1e8, size=n_bootstraps)
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(bsp)(seed) for seed in seeds)
    nulls = np.array(out).T

    sample_t_stats = compute_pop_t_stats(pop_latent)
    for i, sample_t in enumerate(sample_t_stats):
        num_greater = len(np.where(sample_t < nulls[i, :])[0])
        p_val = num_greater / n_bootstraps
        if p_val < 1 / n_bootstraps:
            p_val = 1 / n_bootstraps
        temp.append(p_val)
    node_list.append(temp)
    temp = []

pd.DataFrame(node_list).to_csv(
    savepath / f"null_m{n_graphs}_n{n_verts}_b{n_bootstraps}_s{sims}.csv",
    header=None,
    index=None,
)

