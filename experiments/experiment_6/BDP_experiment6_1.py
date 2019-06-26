#%%
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
import seaborn as sns
from networkx.generators import (
    connected_watts_strogatz_graph,
    barabasi_albert_graph,
    stochastic_block_model,
)

from src.features import (
    avg_clustering_coefficient,
    avg_shortest_path_length,
    degree_assortativity,
    density,
    diameter,
    global_clustering_coefficient,
    node_connectivity,
    num_edges,
    square_clustering,
    triangle_ratio,
)

n_verts = 100
n_graphs = 100
k = 4
p = 0.2
graphs = []

metrics = dict(
    num_edges=num_edges,
    triangle_ratio=triangle_ratio,
    avg_shortest_path_length=avg_shortest_path_length,
    avg_clustering_coefficient=avg_clustering_coefficient,
    global_clustering_coefficient=global_clustering_coefficient,
    diameter=diameter,
    square_clustering=square_clustering,
    degree_assortativity=degree_assortativity,
    density=density,
    node_connectivity=node_connectivity,
)
# generator = connected_watts_strogatz_graph
# args = (n_verts, k, p)
# generator = stochastic_block_model()
# args = (n_verts, 5)
generator = stochastic_block_model
p = np.array([[0.6, 0.1], [0.1, 0.6]])
args = ([int(n_verts / 2), int(n_verts / 2)], p)
graphs_metrics = []
for i in range(n_graphs):
    graph = generator(*args)
    graphs.append(graph)
    graph_metrics = {}
    for name, metric in metrics.items():
        m = metric(graph)
        graph_metrics[name] = m
    graphs_metrics.append(graph_metrics)
graphs_metrics_df = pd.DataFrame(graphs_metrics)

#%%
plt.figure(figsize=(10, 10))
plt.style.use("seaborn-white")
plt.tight_layout()
sns.set_palette("Set1")
sns.set_context("paper", font_scale=1.5)
plt_kws = dict(alpha=0.5, linewidth=0)
sns.pairplot(data=graphs_metrics_df, plot_kws=plt_kws)
plt.savefig("./BDP-10metric-SBM.pdf", facecolor="w", format="pdf")

#%%
plt.figure(figsize=(10, 10))
corr = graphs_metrics_df.corr()
sns.heatmap(corr, annot=True, square=True, cmap="RdBu_r")

