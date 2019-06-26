import networkx as nx
import math


def num_edges(graph):
    return nx.number_of_edges(graph)


def triangle_ratio(graph):
    node_N = nx.number_of_nodes(graph)
    triangle = sum(nx.triangles(graph).values()) / 3
    C_Triangle = math.factorial(node_N) / math.factorial(node_N - 3) / math.factorial(3)
    triangle = float(triangle) / C_Triangle
    return triangle


def avg_shortest_path_length(graph):
    APL = 0
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
        diam = -1
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
