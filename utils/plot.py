import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_matrix(cm: np.ndarray, savepath: str, xtickslabel: list, ytickslabel: list):
    """
    Draw confusion matrix or other.
    """
    size = len(xtickslabel)
    rotation = 0
    fig, ax = plt.subplots(1, 1,figsize = (2*size, 2*size), dpi = 360)
    sns.heatmap(cm, ax = ax, annot = True, fmt = "d", cmap = "Blues")
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predict Label")
    if size is not None and len(size) > 10:
        rotation = 45
    ax.set_xticklabels(xtickslabel, rotation = rotation)
    ax.set_yticklabels(ytickslabel, rotation = rotation)
    plt.savefig(savepath, bbox_inches = "tight")
    plt.close("all")