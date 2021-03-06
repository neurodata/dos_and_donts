import math

import networkx as nx
import numpy as np


def num_edges(graph):
    return nx.number_of_edges(graph)


def is_planar(graph):
    return nx.check_planarity(graph)[0]


def is_connected(graph):
    return nx.is_connected(graph)


def total_triangles(graph):
    return np.sum(list(nx.triangles(graph).values()))


def triangle_ratio(graph):
    node_N = nx.number_of_nodes(graph)
    triangle = sum(nx.triangles(graph).values()) / 3
    C_Triangle = math.factorial(node_N) / math.factorial(node_N - 3) / math.factorial(3)
    triangle = float(triangle) / C_Triangle
    return triangle


def avg_shortest_path_length(graph):
    APL = np.nan
    if nx.number_connected_components(graph) != 1:
        Gc = max(nx.connected_component_subgraphs(graph), key=len)
        if nx.number_of_nodes(Gc) != 1:
            APL = nx.average_shortest_path_length(Gc)
    else:
        APL = nx.average_shortest_path_length(graph)
    return APL


def global_clustering_coefficient(graph):
    GCC = nx.transitivity(graph)
    return GCC


def avg_clustering_coefficient(graph):
    return nx.average_clustering(graph)


def diameter(graph):
    try:
        diam = nx.diameter(graph)
    except:
        diam = np.nan
    return diam


def square_clustering(graph):
    dic = nx.square_clustering(graph).values()
    summation = 0
    for e in dic:
        summation = summation + e
    try:
        summation = summation / len(dic)
    except:
        summation = 0
    return summation


def degree_assortativity(graph):
    try:
        ass = nx.degree_assortativity_coefficient(graph)
    except:
        ass = 0
    return ass


def density(graph):
    return nx.density(graph)


def node_connectivity(graph):
    return nx.node_connectivity(graph)


def global_efficiency(graph):
    return nx.global_efficiency(graph)


def local_efficiency(graph):
    return nx.local_efficiency(graph)


def small_world_omega(graph):
    if nx.number_connected_components(graph) != 1:
        Gc = max(nx.connected_component_subgraphs(graph), key=len)
        if nx.number_of_nodes(Gc) >= 4:
            return nx.omega(Gc, 25)
        else:
            return np.nan
    else:
        return nx.omega(graph, 25)


def small_world_sigma(graph):
    if nx.number_connected_components(graph) != 1:
        Gc = max(nx.connected_component_subgraphs(graph), key=len)
        if nx.number_of_nodes(Gc) >= 4:
            return nx.sigma(Gc, 25)
        else:
            return np.nan
    else:
        return nx.sigma(graph, 25)


def modularity(graph):
    """
    Modularity index based on Clauset-Newman-Moore greedy modularity
    maximization.
    """
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        Q = nx.algorithms.community.quality.modularity(graph, communities)

        return Q
    except:
        # Deal with completely unconnected graph
        return np.nan
