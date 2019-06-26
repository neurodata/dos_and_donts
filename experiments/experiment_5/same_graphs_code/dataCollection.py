import _pickle as cPickle
import os
import networkx as nx

import json
from networkx.readwrite import json_graph

import math
import random
import gc


class DataCollection:
    def __init__(self, name=None, graphs=None):
        # figure out data path and data type
        # 1: do nothing
        # 2: using generator argv[2] will give the order of generator
        # 3: using one file data set such as g6
        # 4: using json data set

        if name is not None and graphs is not None:
            self.dataType = "g6"
            self.graphs = graphs
            self.lens = len(graphs)
            self.saving_name = name
            h, w = 11, self.lens
            self.Matrix = [[0 for x in range(w)] for y in range(h)]
            return
        self.dataType = input(
            "(random_gen,g6,json)what is the experiment you wanna do? "
        )
        if self.dataType == "random_gen":
            self.lens = int(input("Number of Graph for generated: "))
            self.vertex_num = int(input("Number of vertex for generated: "))
            self.genType = input(
                "(WS,BA,ER,geometric)what is the model for this experiment? "
            )
            self.distribute = input("(Y/N)weather to use population distribution? ")
            if self.distribute == "Y":
                self.distribute = True
            else:
                self.distribute = False
            self.saving_name = (
                str(self.genType) + "_" + str(self.lens) + "_" + str(self.vertex_num)
            )
            if self.distribute:
                self.ram_dist = cPickle.load(open("./edge_num_cdf.pkl", "rb"))
                self.ram_dist = self.ram_dist[self.vertex_num - 1]
            # print self.ram_dist
            if self.genType == "WS":
                self.ram_dist = [2] * 771 + [4] * 34040 + [6] * 771 + [8]
            if self.genType == "BA":
                self.ram_dist = (
                    [1] * 172
                    + [8] * 173
                    + [2] * 7808
                    + [7] * 7807
                    + [3] * 17020
                    + [6] * 17020
                    + [4] * 13993
                    + [5] * 13994
                )

        elif self.dataType == "g6":
            file_path = input("please give the file name in dataset folder: ")
            f = open("./dataset/" + file_path, "r")
            if file_path == "graph10.g6":
                self.graphs = f
                self.lens = 12005168
            else:
                self.graphs = nx.read_graph6(f)
                self.lens = len(self.graphs)
            self.saving_name = file_path

        elif self.dataType == "json":
            self.saving_name = input(
                "please give a name for the data of the json experiment "
            )
            file_path = input("what is the json file folder name in dataset folder: ")
            self.dir = os.listdir("./dataset/" + file_path)
            self.lens = 0
            self.dir_list = []
            for dir_path in self.dir:
                f = os.listdir(file_path + "/" + dir_path)
                # print(file_path, dir_path, f)
                for file_name in f:
                    self.dir_list.append(file_path + "/" + dir_path + "/" + file_name)
                # self.graphs += open(file_path+"/"+dir_path+"/"+f,'r')
                self.lens += len(f)
        else:
            print("please give correct input")
            exit(1)

        h, w = 11, self.lens
        self.Matrix = [[0 for x in range(w)] for y in range(h)]

    def run_collection(self):
        percent = -1
        to_write = open("ratio.txt", "w")
        for index in range(self.lens):
            if index % 1000 == 0:
                percent = percent + 1

                gc.collect()
            G = None
            if self.dataType == "random_gen":
                if self.genType == "ER":
                    if self.distribute:
                        G = nx.fast_gnp_random_graph(self.vertex_num, self.get_ran())
                    else:
                        G = nx.fast_gnp_random_graph(self.vertex_num, random.random())
                if self.genType == "WS":
                    if self.distribute:
                        G = nx.watts_strogatz_graph(
                            self.vertex_num, self.get_ran(), random.random()
                        )
                    else:
                        ring_num = random.randint(2, self.vertex_num - 1)
                        G = nx.watts_strogatz_graph(
                            self.vertex_num, ring_num, random.random()
                        )
                if self.genType == "BA":
                    if self.distribute:
                        G = nx.barabasi_albert_graph(self.vertex_num, self.get_ran())
                    else:
                        G = nx.barabasi_albert_graph(
                            self.vertex_num, random.randint(1, self.vertex_num - 1)
                        )
                if self.genType == "geometric":
                    if self.distribute:
                        G = nx.random_geometric_graph(self.vertex_num, self.get_ran())
                    else:
                        G = nx.random_geometric_graph(self.vertex_num, random.random())
            if self.dataType == "g6":
                if self.saving_name == "graph10.g6":
                    word = self.graphs.next()
                    G = nx.from_graph6_bytes(word.rstrip())
                else:
                    G = self.graphs.pop()
            if self.dataType == "json":
                data_file = open(self.dir_list[index], "r")
                data = json.load(data_file)
                g = json_graph.node_link_graph(data)
                # print(str(nx.MultiGraph.selfloop_edges(g)))
                G = nx.Graph()
                for u, v in g.edges():
                    G.add_edge(u, v)
            # print(nx.number_of_nodes(G))
            self.Matrix[0][index] = nx.transitivity(G)
            # print(self.Matrix[0][index])
            try:
                self.Matrix[1][index] = nx.average_clustering(G)
            except:
                self.Matrix[1][index] = 0
            dic = nx.square_clustering(G).values()
            summation = 0
            for e in dic:
                summation = summation + e
            try:
                self.Matrix[2][index] = summation / len(dic)
            except:
                self.Matrix[2][index] = 0
            if nx.number_connected_components(G) != 1:
                # print("dis connecrted")
                Gc = max(nx.connected_component_subgraphs(G), key=len)
                if nx.number_of_nodes(Gc) == 1:
                    continue
                else:
                    self.Matrix[3][index] = nx.average_shortest_path_length(Gc)

                    to_write.write("this is " + str(index) + "'s graph\n")
                    to_write.write(
                        "node ratio: "
                        + str(nx.number_of_nodes(Gc) / nx.number_of_nodes(G))
                        + "\n"
                    )
                    to_write.write(
                        "edge ratio: "
                        + str(nx.number_of_edges(Gc) / nx.number_of_edges(G))
                        + "\n"
                    )
            else:
                self.Matrix[3][index] = nx.average_shortest_path_length(G)
            try:
                self.Matrix[4][index] = nx.degree_assortativity_coefficient(G)
            except:
                self.Matrix[4][index] = 0
            if nx.number_connected_components(G) != 1:
                Gc = max(nx.connected_component_subgraphs(G), key=len)

                self.Matrix[5][index] = nx.diameter(Gc)
            else:
                self.Matrix[5][index] = nx.diameter(G)

            self.Matrix[6][index] = nx.density(G)

            # triangle part, calculate the ratio of triangle
            node_N = nx.number_of_nodes(G)
            if node_N < 3:
                self.Matrix[7][index] = 0
            else:
                triangle = sum(nx.triangles(G).values()) / 3
                C_Triangle = (
                    math.factorial(node_N)
                    / math.factorial(node_N - 3)
                    / math.factorial(3)
                )
                self.Matrix[7][index] = float(triangle) / C_Triangle

            self.Matrix[8][index] = nx.node_connectivity(G)
            self.Matrix[9][index] = nx.edge_connectivity(G)

            egvl = nx.laplacian_spectrum(G)
            summ = 0
            for mu in egvl:
                summ += 1 / mu
            summ = summ / nx.number_of_nodes(G)
            self.Matrix[10][index] = summ
        to_write.close()
        if self.dataType == "random_gen":
            if self.distribute:
                self.saving_name = self.saving_name + "_P"
            else:
                self.saving_name = self.saving_name + "_U"

        cPickle.dump(self.Matrix, open(self.saving_name + ".pkl", "wb"))

    def get_ran(self):
        if self.genType == "ER" or self.genType == "geometric":
            num = float(self.get_random_based_on_population())
            max_v = self.vertex_num * (self.vertex_num - 1) / 2
            if num == 0:
                return 0
            if num == max_v:
                return 1
            return (random.random() - 0.5) / max_v + num / max_v
        return self.get_random_based_on_population()

    def get_random_based_on_population(self):
        if self.genType == "ER" or self.genType == "geometric":
            index = random.randint(1, self.ram_dist[len(self.ram_dist) - 1])
            for i in range(len(self.ram_dist)):
                if index <= self.ram_dist[i]:
                    return i
        index = random.randint(1, len(self.ram_dist))
        return self.ram_dist[index]


if __name__ == "__main__":
    experiment = DataCollection()
    experiment.run_collection()

