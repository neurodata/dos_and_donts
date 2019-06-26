import math
import _pickle as cPickle
import networkx as nx

import drawMatrix
import dataCollection


# the following class is the graph filter,
# its mainly call the function in network and compare them
class GraphFilter:
    def __init__(self, file_path, isNew):
        if isNew:
            f = open(file_path + ".g6", "r")
            self.graphs = nx.read_graph6(f)
        if not isNew:
            self.graphs = cPickle.load(open(file_path + ".pkl", "rb"))

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
            if min_apl < APL < max_apl:
                new_graph.append(G)
        return new_graph

    def GCC_filter(self, min_GCC, max_GCC):
        new_graph = []
        for G in self.graphs:
            GCC = nx.transitivity(G)
            if min_GCC < GCC < max_GCC:
                new_graph.append(G)
        return new_graph

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

    def corr_matrix(self, name):
        temp = self.graphs[:]
        data_c = dataCollection.DataCollection("data_" + name, temp)
        data_c.run_collection()
        drawMatrix.drawMatrix("data_" + name)
        cPickle.dump(self.graphs, open("graph_" + name + ".pkl", "wb"))


#  script, several loops for doing graph filter
while True:
    newAnalysis = input("(g6/pkl)gives a data type: ")
    testData = None
    if "g6" == newAnalysis:
        filePath = input("(g6)gives a file path: ")
        testData = GraphFilter(filePath, True)
        break
    elif "pkl" == newAnalysis:
        filePath = input("(pkl)gives a file path: ")
        testData = GraphFilter(filePath, False)
        break
if testData is None:
    print("ERROR: file does not exist. ")
    exit(1)
to_save = testData.get_graph()
does_save = True
while True:
    print(
        "(APL, regular, eulerian, tree, bipartite, connected, r_tri, edge, diam\n"
        "planar, GCC, ACC, SCC, assort, density, edge/nodeConn, s(save previous operation), \n"
        "draw(save the data as .pkl and draw it), num(check number of Graph left))"
    )
    operation = input("what is the operation for next step: ")
    if operation == "num":
        print(len(to_save))
    if operation == "s":
        testData.set_graph(to_save)
        does_save = True
    if not does_save:
        save_or_not = input(
            "(Y/N)do you wanna save the previous change before doing this experiment? "
        )
        if save_or_not == "Y":
            testData.set_graph(to_save)
        if save_or_not == "N":
            to_save = testData.get_graph()
        does_save = True
    if operation == "draw":
        name = input("please give a name for this set of graphs: ")
        testData.corr_matrix(name)
        ex = input("(Y/N)does end this program?")
        if ex == "Y":
            break
    if operation == "edge":
        give_low = float(input("(float)please give the lower bound of ratio of edge: "))
        give_high = float(
            input("(float)please give the upper bound of ratio of edge: ")
        )
        to_save = testData.edge_filter(give_low, give_high)
        does_save = False
    if operation == "r_tri":
        give_low = float(
            input("(float)please give the lower bound of ratio of triangle: ")
        )
        give_high = float(
            input("(float)please give the upper bound of ratio of triangle: ")
        )
        to_save = testData.triangle_filter(give_low, give_high)
        does_save = False
    if operation == "APL":
        give_low = float(input("(float)please give the lower bound of APL: "))
        give_high = float(input("(float)please give the upper bound of APL: "))
        to_save = testData.APL_filter(give_low, give_high)
        does_save = False
    if operation == "GCC":
        give_low = float(input("(float)please give the lower bound of GCC: "))
        give_high = float(input("(float)please give the upper bound of GCC: "))
        to_save = testData.GCC_filter(give_low, give_high)
        does_save = False
    if operation == "ACC":
        give_low = float(input("(float)please give the lower bound of ACC: "))
        give_high = float(input("(float)please give the upper bound of ACC: "))
        to_save = testData.ACC_filter(give_low, give_high)
        does_save = False
    if operation == "SCC":
        give_low = float(input("(float)please give the lower bound of SCC: "))
        give_high = float(input("(float)please give the upper bound of SCC: "))
        to_save = testData.SCC_filter(give_low, give_high)
        does_save = False
    if operation == "assort":
        give_low = float(
            input("(float)please give the lower bound of degree assortativity: ")
        )
        give_high = float(
            input("(float)please give the upper bound of degree assortativity: ")
        )
        to_save = testData.assort_filter(give_low, give_high)
        does_save = False
    if operation == "density":
        give_low = float(input("(float)please give the lower bound of density: "))
        give_high = float(input("(float)please give the upper bound of density: "))
        to_save = testData.density_filter(give_low, give_high)
        does_save = False
    if operation == "diam":
        give_low = float(input("(float)please give the lower bound of diameter: "))
        give_high = float(input("(float)please give the upper bound of diameter: "))
        to_save = testData.diam_filter(give_low, give_high)
        does_save = False
    if operation == "edgeConn":
        give_low = float(
            input("(float)please give the lower bound of edge connectivity: ")
        )
        give_high = float(
            input("(float)please give the upper bound of edge connectivity: ")
        )
        to_save = testData.edge_connectivity_filter(give_low, give_high)
        does_save = False
    if operation == "nodeConn":
        give_low = float(
            input("(float)please give the lower bound of node connectivity: ")
        )
        give_high = float(
            input("(float)please give the upper bound of node connectivity: ")
        )
        to_save = testData.node_connectivity_filter(give_low, give_high)
        does_save = False
    if operation == "regular":
        is_regular = input("(Y/N)is regular: ")
        if is_regular == "Y":
            to_save = testData.regular(True)
            does_save = False
        if is_regular == "N":
            to_save = testData.regular(False)
            does_save = False
    if operation == "eulerian":
        is_eulerian = input("(Y/N)is eulerian: ")
        if is_eulerian == "Y":
            to_save = testData.eulerian(True)
            does_save = False
        if is_eulerian == "N":
            to_save = testData.eulerian(False)
            does_save = False
    if operation == "tree":
        is_tree = input("(Y/N)is tree: ")
        if is_tree == "Y":
            to_save = testData.tree(True)
            does_save = False
        if is_tree == "N":
            to_save = testData.tree(False)
            does_save = False
    if operation == "bipartite":
        is_bipartite = input("(Y/N)is bipartite: ")
        if is_bipartite == "Y":
            to_save = testData.bipartite(True)
            does_save = False
        if is_bipartite == "N":
            to_save = testData.bipartite(False)
            does_save = False
    if operation == "connected":
        is_connected = input("(Y/N)is connected: ")
        if is_connected == "Y":
            to_save = testData.connected(True)
            does_save = False
        if is_connected == "N":
            to_save = testData.connected(False)
            does_save = False
    if operation == "planar":
        is_planar = input("(Y/N)is planar: ")
        if is_planar == "Y":
            to_save = testData.planar(True)
            does_save = False
        if is_planar == "N":
            to_save = testData.planar(False)
            does_save = False

