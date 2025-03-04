"""
    Authors:
        Niraj Bhujel (niraj.bhujel@stfc.ac.uk) SciML, SCD-STFC
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import sklearn.metrics as skmetrics 

def _binary_stat_scores(preds, target):
    """
    Compute the statistics. Modified from torchmetrics.classification.stat_scores.py
    Inputs:
        preds: (B, 1), pred class indexes
        target: (B, 1), ground truth class indexes 
    """
    tp = ((target == preds) & (target == 1))
    fn = ((target != preds) & (target == 1))
    fp = ((target != preds) & (target == 0))
    tn = ((target == preds) & (target == 0))
    
    return tp, fp, tn, fn


class ClassificationMetricLogger:
    def __init__(self, name="metric_logger", num_class=2, class_labels=None, threshold=0.5, trace_func=print):

        self.name = name
        self.num_class = num_class
        self.class_labels = class_labels
        self.threshold = threshold
        self.trace_func = trace_func
        self.reset()

    def __len__(
        self,
    ):
        return self.count

    def get_class_labels(self, ):
        if self.class_labels is not None:
            return self.class_labels
        else:
            return np.arange(self.num_class)
    @property
    def binary_classification(self, ):
        return False if self.num_class>2 else True

    @property
    def multiclass_classification(self, ):
        return True if self.num_class>2 else False
        
    def reset(
        self,
    ):
        self.count = 0
        self.results = {}
        self.y_true = [] # true class label either 0 or 1
        self.y_pred = [] # pred class label either 0 or 1
        self.y_logits = []
        self.y_prob = [] # probability of predicted class label 0 to 1
        self.files = []
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        self.best_results = None

    def _validate(self, y_logits, y_true):

        assert y_true.ndim<=2, f"y_logits shape should be either (B, C) or (B, ), got {y_true.shape=}"
        assert y_logits.ndim<=2, f"y_logits shape should be either (B, C) or (B, ), got {y_logits.shape=}"

        if y_logits.ndim==2:
            assert y_logits.shape[-1]==self.num_class, f"Size mismatch at last dimension in y_logits, got {y_logits.shape[-1]=} instead of {self.num_class}"
            
    def __call__(self, y_logits, y_true, files=[]):

        """
        Inputs:
            y_true (np.array | torch.tensor | list): Ground truth label, either onhot encoded (B, C) or categorical (B, ) 
            y_logits (np.array | torch.tensor | list): Assume to be output from softmax, either (B, C) or (B, ). 
        """
        self.count += 1

        assert len(y_logits)==len(y_true), f"Shape mismatch at dimension 0, got {y_logits.shape=}, {y_true.shape=}"

        if isinstance(y_true, list):
            y_true = np.asarray(y_true)
            
        if isinstance(y_logits, list):
            y_logits = np.asarray(y_logits)
            
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
            
        if isinstance(y_logits, torch.Tensor):
            y_logits = y_logits.cpu().numpy()

        self._validate(y_logits, y_true)
        
        if y_true.ndim==2:
            y_true = y_true.argmax(-1)

        if y_logits.ndim==2:
            y_prob = y_logits.max(-1)
        else:
            y_prob = y_logits

        if self.binary_classification:
            y_pred = (y_logits>self.threshold)
            if y_logits.ndim>1:
                y_pred = y_pred.argmax(-1)

        else:
            if y_logits.ndim>1:
                y_pred = y_logits.argmax(-1)
            else:
                y_pred = y_logits

        self.y_logits.extend(y_logits.tolist())
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())
        self.y_prob.extend(y_prob.tolist())
        self.files.extend(files)

        acc = (y_true == y_pred).astype(np.float32)
        
        return acc

    def compute(self, tabulate_results=True):
        
        if not len(self) > 0:
            raise Exception("Nothing to compute. Please update results first")
        # Compute Confusion Matrix
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            self.confusion_matrix[true_label, pred_label] += 1
            
        N = np.sum(self.confusion_matrix)

        # Extract TP, FP, FN, TN
        if self.multiclass_classification:
            tp = np.diag(self.confusion_matrix)
            fp = np.sum(self.confusion_matrix, axis=0) - tp
            fn = np.sum(self.confusion_matrix, axis=1) - tp
            tn = np.sum(self.confusion_matrix) - (tp + fp + fn)
        else:
            tp = self.confusion_matrix[1, 1]
            fp = self.confusion_matrix[0, 1]
            fn = self.confusion_matrix[1, 0]
            tn = self.confusion_matrix[0, 0]

        accuracy = (tp + tn )/N
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)>0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)>0)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(precision), where=(precision+recall)>0)
        
        if self.multiclass_classification:
            self.results = dict(
                tp=np.sum(tp), 
                fp=np.sum(fp), 
                tn=np.sum(tn),
                fn=np.sum(fn), 
                accuracy=np.mean(accuracy), 
                precision=np.mean(precision), 
                recall=np.mean(recall), 
                f1=np.mean(f1),
            )
        else:
            self.results = dict(
                tp=tp, 
                fp=fp, 
                tn=tn,
                fn=fn, 
                accuracy=accuracy, 
                precision=precision, 
                recall=recall, 
                f1=f1,
            )

        if tabulate_results:
            self.trace_func(self.tabulate_metrics())
        
        # self.print_metrics_table(self.trace_func)
        
    def tabulate_metrics(self, tablefmt="outline", floatfmt=".3f", col_width=0):
        """
        Tabluate the results dictionary.
        """
        metrics = self.results
        
        headers = [f"{self.name:<}"] + [f"{h:^}" for h in metrics.keys()]
        separator = ['-'*(len(h)+2) for h in headers]
        table = []

        # if self.multiclass_classification:
        #     for i in range(self.num_class):
        #         cls_label = self.get_class_labels()[i]
        #         cls_row = [f"{f'{cls_label}':<}"]
        #         for k, v in metrics.items():
        #             cls_row.append(f"{int(v[i])}" if k in ['tp', 'fp', 'tn', 'fn'] else f"{v[i]:{floatfmt}}")
        #             # cls_row.append(int(v[i]) if k in ['tp', 'fp', 'tn', 'fn'] else v[i]) 
                    
        #         table.append(cls_row)
        #     table.append(separator)
            
        avg_row = [f"{'Average':<{col_width}}"]
        for k, v in metrics.items():
            # avg_row.append(f"{int(np.sum(v))}" if k in ['tp', 'fp', 'tn', 'fn'] else f"{np.mean(v):{floatfmt}}") 
            avg_row.append(f"{int(v)}" if k in ['tp', 'fp', 'tn', 'fn'] else f"{v:{floatfmt}}") 

        table.append(avg_row)
        tabular_data = tabulate(table, headers=headers, tablefmt=tablefmt)

        return tabular_data
        
    @property
    def df(self, ):
        data_dict = dict(y_true=self.y_true, 
                         y_pred=self.y_pred, 
                         y_prob=self.y_prob, 
                         files=self.files,
                        )
        return pd.DataFrame(data_dict)
        
    def plot_confusion_matrix(self, cmap='Blues', cbar=False, fontsize=16, figsize=(5,5)):
        
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        
        sns.heatmap(self.confusion_matrix, 
                    annot=True, 
                    annot_kws = {'fontsize': fontsize},
                    fmt='g', 
                    linewidth=.5, 
                    ax=ax, 
                    cmap=cmap, 
                    cbar=cbar,
                    )
        
        ax.set(
            xticks=np.arange(self.num_class)+0.5,
            yticks=np.arange(self.num_class)+0.5,
            xticklabels=self.get_class_labels(),
            yticklabels=self.get_class_labels(),
        )
        # Set font size for tick labels
        ax.tick_params(axis="both", labelsize=fontsize)  # Adjust 12 to desired font size


        ax.set_xlabel("Predicted label", fontsize=fontsize)
        ax.set_ylabel("True label", fontsize=fontsize)

        # set title
        ax.set_title(self.name, fontsize=fontsize)
        
        # Color x_ticks/y_ticks labels
        colors = sns.color_palette('bright', n_colors=self.num_class)
        [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
        [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # plt.show()
    
    def plot_precision_recall(self, ):
        # precision, recall, _ = skmetrics.precision_recall_curve(self.y_true, self.y_pred)
        skmetrics.PrecisionRecallDisplay(precision=self.results["precision"], 
                                         recall=self.results["recall"]
                                        ).plot()

    def plot_roc_curve(self, ):
        fpr, tpr, thresholds = skmetrics.roc_curve(self.y_true, self.y_pred)
        roc_auc = skmetrics.auc(fpr, tpr)
        skmetrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()

    @property
    def roc_auc_score(self, ):
        return skmetrics.roc_auc_score(self.y_true, self.y_pred)
