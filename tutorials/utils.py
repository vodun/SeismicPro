''' Utility funcitons for tutorials '''

import matplotlib.pyplot as plt

def histogram_plotter(gather_list, bins=41):
    ''' Plot data historgams for each gather in `gather_list` '''
    n_plots = len(gather_list)
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    for i in range(n_plots):
        ax[i].set_title(gather_list[i][1])
        ax[i].set_xlabel('values')
        ax[i].set_ylabel('counts')
        _ = ax[i].hist(gather_list[i][0].data.ravel(), bins=bins)
