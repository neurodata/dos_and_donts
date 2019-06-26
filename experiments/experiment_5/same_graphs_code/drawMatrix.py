import _pickle as cPickle
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import sys

# you can change configuration, the code will load and draw the first path without extension(./data/graph9
# ./data/graph8) will draw 2 matrix based on graph9 and graph8 from data folder


class drawMatrix:
    def __init__(self, lists=None):
        self.graph10 = False
        self.color_f_plot = ["b", "r", "g", "c", "m", "y", "k"]
        self.stats_label = [
            "GCC",
            "ACC",
            "SCC",
            "APL",
            "r",
            "diam",
            "den",
            "Rt",
            "Cv",
            "Ce",
            "E_G_resist",
        ]
        self.my_drawing_list = []
        if lists is not None:
            self.my_drawing_list.append(lists)
        else:
            if len(sys.argv) >= 2:
                my_drawing_list = sys.argv[1 : len(sys.argv)]
                self.draw_multi_graph(my_drawing_list)

        self.draw_multi_graph(self.my_drawing_list)

    def draw_multi_graph(self, my_drawing_list):
        # plt.figure(0)
        yMin = []
        yMax, xMax = [], []
        for times in range(len(my_drawing_list)):
            if times == 0 and self.graph10:
                continue
            filename = my_drawing_list[times]
            Matrix = cPickle.load(open(filename + ".pkl", "rb"))
            Matrix = Matrix[0:10]
            assort = Matrix[4]
            for x in range(len(assort)):
                if np.isnan(assort[x]):
                    assort[x] = 0
            Matrix[4] = assort
            index = 0
            for i in range(len(Matrix)):
                list1 = Matrix[i]

                for j in range(len(Matrix)):
                    list2 = Matrix[j]

                    if j < i:
                        continue
                    init = 0
                    if self.graph10:
                        init = 1
                    if times == init:
                        yMin.append(min(list1))
                        yMax.append(max(list1))
                        xMax.append(max(list2))
                    else:
                        if min(list1) < yMin[index]:
                            yMin[index] = min(list1)
                        if max(list1) > yMax[index]:
                            yMax[index] = max(list1)
                        if max(list2) > xMax[index]:
                            xMax[index] = max(list2)
                    index = index + 1

        for times in range(len(my_drawing_list)):
            filename = my_drawing_list[times]
            Matrix = cPickle.load(open(filename + ".pkl", "rb"))
            Matrix = Matrix[0:10]
            assort = Matrix[4]
            for x in range(len(assort)):
                if np.isnan(assort[x]):

                    assort[x] = 0
            Matrix[4] = assort

            print("number of graph: " + str(len(Matrix[0])))
            print(filename)
            index = 0
            for i in range(len(Matrix)):
                list1 = Matrix[i]

                for j in range(len(Matrix)):
                    list2 = Matrix[j]

                    if j < i:
                        continue

                    plt.subplot(len(Matrix), len(Matrix), j + i * len(Matrix) + 1)
                    if i == j:
                        plt.xlabel(self.stats_label[j], fontsize=20)
                        plt.ylabel(self.stats_label[i], fontsize=20)
                    corr, p = stats.pearsonr(list1, list2)
                    if np.isnan(corr):
                        print("p value for " + str(i) + " " + str(j) + ": " + str(p))
                    yscale = (yMax[index] - yMin[index]) / 2
                    plt.text(
                        xMax[index] + 0.05,
                        yMax[index] - times * 0.5 * yscale,
                        "{0:.2f}".format(corr),
                        color=self.color_f_plot[times],
                    )
                    if i == len(Matrix) - 1 and j == len(Matrix) - 1:
                        plt.text(
                            -14 * xMax[index],
                            yscale * 10 - times * yscale * 1.6,
                            filename,
                            color=self.color_f_plot[times],
                            fontsize=20,
                        )
                    #     filename.lstrip('.v9/dataset')
                    if self.graph10:
                        if times != 0:
                            plt.plot(list2, list1, self.color_f_plot[times] + "o")
                    else:
                        plt.plot(list2, list1, self.color_f_plot[times] + "o")

                    # plt.plot(list2, list1, self.color_f_plot[times] + "o")
                    index = index + 1
                    # plt.plot(list2,list1,color_f_plot[times])
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.subplots_adjust(
            left=0.03, bottom=0.07, right=0.97, top=0.98, hspace=0.5, wspace=0.47
        )
        plt.show()
        # plt.savefig("gd5-10.png",dpi=300)


def data_to_log(list_to_modify):
    for ii in range(len(list_to_modify)):
        if list_to_modify[ii] == 0:
            continue
        list_to_modify[ii] = math.log(list_to_modify[ii])

    return list_to_modify


if __name__ == "__main__":
    graph = drawMatrix()
