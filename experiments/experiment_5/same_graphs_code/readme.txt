please use python 2.7.14, If you use a non-windows machine, please delete line 113,114 in drawMatrix.py. if you cannot install planarity properly, just delete this function or install Cython

This codes have three part, dataCollection.py(data processing), drawMatrix.py(visualize the data through matrix plot) and graph_filter.py(generating many graphs that are identical over a number of graph properties and statistics yet are clearly different and identifiably distinct.).

For the most important part, graph_filter.py. you can just run it. it will ask you questions. Answering all the task you wanna do. you will get 2 files, data_XXX.pkl and graph_XXX.pkl. you can add data_XXX in the configuration of drawMatrix.py and see the plot again. For graph_XXX.pkl, you can continue doing filter thing by graph_filter. graph_XXX.pkl is a list of networkX Graph. you can also do you own experiment from there. 

For dataCollection.py, you can also answer question and get the data for drawing(drawMatrix.py)

For drawMatrix.py, you will NOT answer question. please put the thing you want to plot in the configuration. this code will draw them in order from first to the end.