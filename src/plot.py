import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_heatmap(data, xlabels, ylabels):
    sns.set_context("talk", font_scale=1.25)
    fig, ax = plt.subplots()
