import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ORDERING = dict(
    # num_edges = "# Edges",
    density="Density",
    # total_triangles = '# Triangles',
    triangle_ratio="Triangle Ratio",
    is_planar="Is Planar Graph?",
    avg_shortest_path_length="Avg Shortest Path",
    global_clustering_coefficient="Global Clustering",
    avg_clustering_coefficient="Avg Clustering",
    square_clustering="Square Clustering",
    global_efficiency="Global Efficiency",
    local_efficiency="Local Efficiency",
    # degree_assortativity = "Degree Assortativity",
    # diameter = 'Diameter',
    node_connectivity="Node Connectivity",
    modularity="Modularity",
)


def hexbin(
    df,
    ordering,
    x_col="modularity",
    figsize=(18, 16),
    gridsize=40,
    cmap="Blues",
    bins="log",
    title=None,
    savefig=None,
):
    ncols = 3
    nrows = int(np.ceil(len(ordering) / ncols))

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharex=True)
    ax = ax.ravel()

    for idx, (col, y_label) in enumerate(ordering.items()):
        ax[idx].hexbin(
            x=df.modularity, y=df[col], cmap=cmap, gridsize=gridsize, bins=bins
        )
        sns.despine(ax=ax[idx])
        # ax[idx].set_xlabel('Modularity', fontsize=20)
        ax[idx].set_ylabel(y_label, fontsize=20)

    for i in range(1, 4):
        ax[-i].set_xlabel(ordering[x_col], fontsize=20)

    if len(ordering) != (ncols * nrows):
        for i in range(1, len(ax) - len(ordering) + 1):
            fig.delaxes(ax[-i])

    if title is not None:
        ax[1].set_title(title, fontsize=30)

    fig.tight_layout()

    if savefig is not None:
        fig.savefig(f"{savefig}.png", dpi=300)

    plt.close()
