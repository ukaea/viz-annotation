
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def plot_bar_metrics(
    metrics_list,
    metrics_to_plot=['accuracy', 'precision', 'recall', 'f1'],
    figtitle='Classification', 
    xticks_label='Iteration', 
    bar_width=0.2, 
    cmap='tab10', 
    ax=None,
    figsize=(8, 6),
    fontsize=8,
    save_dir=None,
):
    '''
    Plot bar chart of the Accuracy and FPs/TPs
    metrics_list (list): List of dictionary for each experiment. 
    '''
    
    N = len(metrics_list)
    
    metric_dict = defaultdict(list)
    for metrics in metrics_list:
        for k, v in metrics.items():
            metric_dict[k].append(v)
        
    # Plot metric results over iteration
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    X = np.arange(N)
    
    multiplier = 0
    colors = plt.get_cmap(cmap)
    for i, m in enumerate(metrics_to_plot):
        m_values = [np.round(v, 2) for v in metric_dict[m]]
        
        offset = bar_width * multiplier
        multiplier += 1
        rects = ax.bar(X + offset, m_values, bar_width, color=colors(i), label=m.title())
        ax.bar_label(rects, padding=3, fontsize=fontsize)
        
    ax.set_title(f"ELM {figtitle}")
    ax.legend(loc='upper left', ncols=1, fontsize=fontsize, framealpha=0.4)
    ax.set_xticks(X + bar_width, labels=[f"{xticks_label}{i+1}" for i in range(N)])
    ax.set_yticks([])
       

    if save_dir is not None:
        plt.savefig(f"{save_dir}/{figtitle}.png")
        
    return ax

