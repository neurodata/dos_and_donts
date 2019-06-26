#%% for jupyter/vs code compatibility
import os

import networkx as nx
import numpy as np
from networkx.generators import (
    balanced_tree,
    barbell_graph,
    circulant_graph,
    circular_ladder_graph,
    complete_graph,
    connected_caveman_graph,
    connected_watts_strogatz_graph,
    cycle_graph,
    erdos_renyi_graph,
    full_rary_tree,
    ladder_graph,
    lollipop_graph,
    newman_watts_strogatz_graph,
    star_graph,
    turan_graph,
    random_shell_graph,
    random_lobster,
    random_powerlaw_tree,
    random_partition_graph,
    stochastic_block_model,
)

from graph_filter import GraphFilter

try:
    os.chdir(os.path.join(os.getcwd(), "dos_and_donts/experiments/experiment_5"))
    print(os.getcwd())
except:
    pass

import seaborn as sns
import matplotlib.pyplot as plt

import math

random_graph_gens = [
    erdos_renyi_graph,
    connected_watts_strogatz_graph,
    connected_caveman_graph,
    random_shell_graph,
    random_lobster,
    random_powerlaw_tree,
    stochastic_block_model,
    random_partition_graph,
]

random_graph_args = [(0.3), (5, 0.3)]

n_verts = 50
n_graphs = 1000


def gen_graphs(generators, args, n_verts, n_graphs):
    graphs = []
    for g, a in zip(generators, args):
        for n in range(n_graphs):
            if isinstance(a, tuple):
                graph = g(n_verts, *a)
            else:
                graph = g(n_verts, a)
            graphs.append(graph)
    return graphs


graphs = gen_graphs(random_graph_gens[:2], random_graph_args[:2], n_verts, n_graphs)
gf = GraphFilter(graphs)

# Global clustering coefficient
remaining, scores = gf.GCC_filter(0, 100)
plt.figure()
sns.distplot(scores)
plt.xlabel("GCC")

# Average path length
remaining, scores = gf.APL_filter(0, 20)
plt.figure()
sns.distplot(scores)
plt.xlabel("GCC")


def num_edges(graph):
    return nx.number_of_edges(graph)


def triangles(graph):
    node_N = nx.number_of_nodes(graph)
    triangle = sum(nx.triangles(graph).values()) / 3
    C_Triangle = math.factorial(node_N) / math.factorial(node_N - 3) / math.factorial(3)
    triangle = float(triangle) / C_Triangle
    return triangle


def APL(graph):
    APL = 0
    if nx.number_connected_components(graph) != 1:
        Gc = max(nx.connected_component_subgraphs(graph), key=len)
        if nx.number_of_nodes(Gc) == 1:
            continue
        else:
            APL = nx.average_shortest_path_length(Gc)
    else:
        APL = nx.average_shortest_path_length(graph)
    return APL


def GCC(graph):
    GCC = nx.transitivity(graph)
    return GCC


def ACC_filter(self, min_ACC, max_ACC):
    new_graph = []
    for G in self.graphs:
        ACC = nx.average_clustering(G)
        if min_ACC < ACC < max_ACC:
            new_graph.append(G)
    return new_graph


def diam_filter(self, min_diam, max_diam):
    new_graph = []
    for G in self.graphs:
        try:
            diam = nx.diameter(G)
        except:
            diam = -1
        if min_diam < diam < max_diam:
            new_graph.append(G)
    return new_graph


def SCC_filter(self, min_SCC, max_SCC):
    new_graph = []
    for G in self.graphs:
        # print(nx.square_clustering(G))
        dic = nx.square_clustering(G).values()
        summation = 0
        for e in dic:
            summation = summation + e
        try:
            summation = summation / len(dic)
        except:
            summation = 0

        if min_SCC < summation < max_SCC:
            new_graph.append(G)
    return new_graph


def assort_filter(self, min_ass, max_ass):
    new_graph = []
    for G in self.graphs:
        try:
            ass = nx.degree_assortativity_coefficient(G)
        except:
            ass = 0
        if min_ass < ass < max_ass:
            new_graph.append(G)
    return new_graph


def density_filter(self, min_den, max_den):
    new_graph = []
    for G in self.graphs:
        den = nx.density(G)
        if min_den < den < max_den:
            new_graph.append(G)
    return new_graph


def node_connectivity_filter(self, min_node_connectivity, max_node_connectivity):
    new_graph = []
    for G in self.graphs:
        node_connectivity = nx.node_connectivity(G)
        if min_node_connectivity < node_connectivity < max_node_connectivity:
            new_graph.append(G)
    return new_graph


def edge_connectivity_filter(self, min_edge_connectivity, max_edge_connectivity):
    new_graph = []
    for G in self.graphs:
        edge_connectivity = nx.edge_connectivity(G)
        if min_edge_connectivity < edge_connectivity < max_edge_connectivity:
            new_graph.append(G)
    return new_graph


def connected(self, connected):
    new_graph = []
    for G in self.graphs:
        if nx.is_connected(G) == connected:
            new_graph.append(G)
    return new_graph


def bipartite(self, bipa):
    new_graph = []
    for G in self.graphs:
        if bipa == nx.is_bipartite(G):
            new_graph.append(G)
    return new_graph


def tree(self, tree):
    new_graph = []
    for G in self.graphs:
        if nx.is_tree(G) == tree:
            new_graph.append(G)
    return new_graph


def eulerian(self, euler):
    new_graph = []
    for G in self.graphs:
        if euler == nx.is_eulerian(G):
            new_graph.append(G)
    return new_graph


def regular(self, regul):
    new_graph = []
    for G in self.graphs:
        if regul == self.is_regular(G):
            new_graph.append(G)
    return new_graph


def is_regular(self, graph):
    ll = nx.degree(graph)
    max_num, min_num = 0, 0
    for node in ll:
        num = node[1]
        if num > max_num:
            max_num = num
        if num < min_num:
            min_num = num
    return max_num == min_num


metrics = [APL, GCC, triangles]


class GraphGenerator:
    def __init__(self, generator, prior_funcs, prior_args, metrics, target_graph):
        self.generator = generator
        self.prior_funcs = prior_funcs
        self.prior_args = prior_args
        self.target_graph

    def fit(self, graph):
        target_metrics = []
        for metric in self.metrics:
            m = metric(self.target_graph)
            target_metrics.append(m)

