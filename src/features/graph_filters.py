import networkx as nx


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
