import numpy as np
import matplotlib.pyplot as plt


##############################
#                            #
#         Utilities          #
#                            #
##############################

def unpack(x):
    if x:
        return x[0]
    return np.nan


def create_grouped_bar_plot(group_labels, **kwargs):
    bar_width = 0.15
    num_groups = len(kwargs)

    # Create a bar plot
    fig, ax = plt.subplots()
    x = np.arange(len(group_labels))
    i = 0
    for key, value in kwargs.items():
        ax.bar(x + i*bar_width - (num_groups-1)/2*bar_width, value, bar_width, label=key)
        i += 1

    # Add labels and a legend
    ax.set_ylabel('IG Values')
    ax.axhline(y=0, c="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45)
    ax.legend()

    plt.show()
