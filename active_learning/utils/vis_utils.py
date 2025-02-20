
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def plot_bar_metrics(
    metrics_list, 
    metric_type='classification', 
    xticks_label='Iteration', 
    bar_width=0.2, 
    cmap='tab10', 
    figsize=(10, 6),
    fontsize=8,
):
    '''
    Plot bar chart of the Accuracy and FPs/TPs
    metrics_list (list): List of dictionary
    '''
    
    N = len(metrics_list)
    
    metric_dict = defaultdict(list)
    for metrics in metrics_list:
        for k, v in metrics.items():
            metric_dict[k].append(v)
        
    # Plot metric results over iteration
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    
    X = np.arange(N)
    
    multiplier = 0
    colors = plt.get_cmap(cmap)
    for i, m in enumerate(['accuracy', 'precision', 'recall']):
        m_values = [np.round(v, 2) for v in metric_dict[m]]
        
        offset = bar_width * multiplier
        multiplier += 1
        rects = axes[0].bar(X + offset, m_values, bar_width, color=colors(i), label=m)
        axes[0].bar_label(rects, padding=3, fontsize=fontsize)
        
    axes[0].set_title(f"ELM {metric_type} vs. Iterations")
    axes[0].legend(loc='upper left', ncols=1, fontsize=fontsize, framealpha=0.6)
    axes[0].set_xticks(X + bar_width, labels=[f"{xticks_label}{i+1}" for i in range(N)])
       
    for i, m in enumerate(['tp', 'tn']):
        m_values = [round(v, 2) for v in metric_dict[m]]
        
        offset = bar_width * multiplier
        multiplier += 1
        rects = axes[1].bar(X + offset, m_values, bar_width, color=colors(i), label=m)
        axes[1].bar_label(rects, padding=3)
        
    axes[1].set_title(f"ELM {metric_type} vs. Iteration")
    axes[1].legend(loc='upper left', ncols=1, fontsize=fontsize, framealpha=0.6)
    axes[1].set_xticks(X + bar_width, labels=[f"{xticks_label}{i+1}" for i in range(N)])

    plt.show()

