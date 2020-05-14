import numpy as np
import matplotlib.pyplot as plt

def plot_metric(metric_name, dataset_labels, classifier_metrics):
    # width of the bars
    barWidth = 0.3

    for means, confidence_intervals, classifier in classifier_metrics:
        yer = list(map(lambda cf: (cf[1] - cf[0]) / 2, confidence_intervals))
        r = np.arange(len(means))

        plt.bar(r, means, width = barWidth, edgecolor = 'black', yerr=yer, capsize=7, label=classifier)

    # general layout
    total_metrics = len(dataset_labels)

    #plt.xticks([r + (barWidth * (total_metrics - 1)) for r in range(total_metrics)], dataset_labels)
    plt.xticks(list(range(total_metrics)), dataset_labels)
    plt.ylim(( 0, 1 ))
    plt.ylabel(metric_name)
    plt.legend()

    # Show graphic
    plt.show()
