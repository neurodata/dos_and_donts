import math
import networkx as nx


# the following class is the graph filter,
# its mainly call the function in network and compare them
class GraphFilter:
    def __init__(self, graphs):
        if isinstance(graphs, list):
            self.graphs = graphs
        else:
            raise TypeError("Did not input a list of graphs")

    def get_graph(self):
        return self.graphs

    def number_of_graph(self):
        return len(self.graphs)

    def set_graph(self, graph):
        self.graphs = graph

    def edge_filter(self, min_edge, max_edge):
        new_graph = []
        for G in self.graphs:
            edge = nx.number_of_edges(G)
            node = nx.number_of_nodes(G)
            edge = float(edge) / float(node * (node - 1) / 2)
            if min_edge < edge < max_edge:
                new_graph.append(G)
        return new_graph

    def triangle_filter(self, min_triangle, max_triangle):
        new_graph = []
        for G in self.graphs:
            node_N = nx.number_of_nodes(G)
            triangle = sum(nx.triangles(G).values()) / 3
            C_Triangle = (
                math.factorial(node_N) / math.factorial(node_N - 3) / math.factorial(3)
            )
            triangle = float(triangle) / C_Triangle
            if min_triangle < triangle < max_triangle:
                new_graph.append(G)
        return new_graph

    def APL_filter(self, min_apl, max_apl):
        new_graph = []
        scores = []
        for G in self.graphs:
            APL = 0
            if nx.number_connected_components(G) != 1:
                Gc = max(nx.connected_component_subgraphs(G), key=len)

                if nx.number_of_nodes(Gc) == 1:
                    continue
                else:
                    APL = nx.average_shortest_path_length(Gc)
            else:
                APL = nx.average_shortest_path_length(G)
            scores.append(APL)
            if min_apl < APL < max_apl:
                new_graph.append(G)
        return new_graph, scores

    def GCC_filter(self, min_GCC, max_GCC):
        new_graph = []
        scores = []
        for G in self.graphs:
            GCC = nx.transitivity(G)
            scores.append(GCC)
            if min_GCC < GCC < max_GCC:
                new_graph.append(G)
        return new_graph, scores

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

    # def planar(self, val):
    #     new_graph = []
    #     for G in self.graphs:
    #         if planarity.is_planar(G) == val:
    #             new_graph.append(G)
    #     return new_graph

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

    # def corr_matrix(self, name):
    #     temp = self.graphs[:]
    #     data_c = dataCollection.DataCollection("data_" + name, temp)
    #     data_c.run_collection()
    #     drawMatrix.drawMatrix("data_" + name)
