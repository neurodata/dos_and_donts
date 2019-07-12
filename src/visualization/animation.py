import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


LABELS = dict(
    # num_edges = "# Edges",
    modularity="Modularity",
    density="Density",
    # total_triangles = '# Triangles',
    triangle_ratio="Triangle Ratio",
    # is_planar="Is Planar Graph?",
    avg_shortest_path_length="Avg Shortest Path",
    global_clustering_coefficient="Global Clustering",
    avg_clustering_coefficient="Avg Clustering",
    # square_clustering="Square Clustering",
    global_efficiency="Global Efficiency",
    local_efficiency="Local Efficiency",
    # degree_assortativity="Degree Assortativity",
    # diameter = 'Diameter',
    node_connectivity="Node Connectivity",
)

POSITION = nx.circular_layout(range(0, 10))
SPACING = 0.125
FONTDICT = {"family": "monospace", "weight": "normal", "size": 30}


def make_frame(graph, data, ax):
    """
    graph = nx.Graph
    data = pd.Series
    """
    # font = FontProperties()
    # font.set_family('monospace')

    # fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(graph, pos=POSITION, ax=ax)

    # Dealing with variable values
    # values we plot are based on LABELS variable
    x_pos = 1.2
    loc = (data.size * SPACING) / 2
    y_pos = np.linspace(loc, -loc, len(LABELS))

    max_char = max([len(name) for _, name in LABELS.items()])

    for idx, (key, name) in enumerate(LABELS.items()):
        value = data[key]
        name = name.ljust(max_char) + ": "

        if not np.issubdtype(value.dtype, np.bool_):
            text = name + "{: .9f}".format(value)
            ax.text(x_pos, y_pos[idx], text, fontdict=FONTDICT, alpha=0.3)
            ax.text(x_pos, y_pos[idx], text[:-7], fontdict=FONTDICT, alpha=1)
        else:
            text = f"{name} {value}"
            ax.text(x_pos, y_pos[idx], text, fontdict=FONTDICT, alpha=1)


def make_gif(graphs, df, name="visualization.gif"):
    indices = df.index
    graphs_subset = [graphs[i] for i in indices]

    fig, ax = plt.subplots(figsize=(21, 10))

    def update(i):
        ax.clear()

        idx = indices[i]
        g = graphs_subset[idx]
        data = t.loc[idx]

        make_frame(g, data, ax)
        # plt.tight_layout()

    ani = FuncAnimation(
        fig, update, interval=100, frames=range(t.shape[0]), repeat=True
    )
    ani.save(name, writer="imagemagick", savefig_kwargs={"facecolor": "white"}, fps=16)

    plt.close()
