import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib import cm
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


# defining a function to compute BFs for differences with chance levels for a group of decoding accuracies over time
def bf_testing_time_decod(scores, bf, plot_title="Sensor space decoding", chance=0.5):
    
    # sanity check (these should be the same)
    # print("shape of aggregated scores:", scores.shape, "shape of BFs:", bf.shape)

    # returning an error message if condition is not met
    assert scores.shape == bf.shape, "scores and bf objects should have the same shape"

    # converting scores to a dataframe [this should be participants x timepoints accuracy matrix]
    # df = pd.DataFrame(scores)
    
    # retrieve the number of timepoints
    # n_timepoints = df.shape[1]

    # defining the grid
    fig, axs = plt.subplots(2, figsize=(12, 6))
    
    # defining the main title
    fig.suptitle(plot_title, size=14, weight=800)
    # set_title

    # defining the x-axis ticks
    x_ticks=list(range(1, scores.shape[1]+1))
    
    # plotting the chance level
    axs[0].axhline(y=chance, color="k", ls="--", label="Chance level")
    
    # plotting the stimulus onset
    axs[0].axvline(x=0, color="k", ls="-")
    
    # plotting the accuracy level over time
    # axs[0].plot(decoding_epochs.times, np.mean(scores, axis=0))
    # plotting the average decoding accuracy (over folds) with 95% confidence interval
    # reshaping these results in a pandas dataframe
    results_df = pd.DataFrame(scores.transpose())
    results_df["time"] = x_ticks
    results_df = pd.melt(results_df, id_vars="time", var_name="fold", value_name="score")
    results_df.head()
    
    sns.lineplot(data=results_df, x="time", y="score", ax=axs[0], lw=2)
    
    # axes aesthetics
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_ylabel("Decoding accuracy",fontsize=12)
    axs[0].set_xlabel("")
    
    bf_cols = sns.color_palette("coolwarm", 500)
    exponential_minmax = 3
    val_col_map = np.logspace(-exponential_minmax,exponential_minmax,num=500)
    
    x = x_ticks
    y = bf
    markerline, stemlines, baseline = axs[1].stem(x, y, bottom=1,linefmt="k", markerfmt=None, basefmt=None)
    
    markerline.set_markerfacecolor("w")
    markerline.set_markeredgecolor("w")
    baseline.set_color("k")
    stemlines.set_linewidth(0.5)
    
    cols_idx = [np.argmin(np.abs(val_col_map-i)) for i in y]  
    
    # plotting the stimulus onset
    axs[1].axvline(x=0, color="k", ls="-")
    [axs[1].plot(x[i],y[i],color=bf_cols[cols_idx[i]], marker=".", markersize=8,lw=0,markeredgecolor=None) for i in range(len(cols_idx))]
    axs[1].set_yscale("log")
    axs[1].set_ylim([3**-exponential_minmax,3**exponential_minmax])
    axs[1].set_yticks([1.e-3, 1.e-2, 1, 1.e+2, 1.e+3])
    axs[1].get_xaxis().get_major_formatter().labelOnlyBase = True
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].set_xlabel("Time (s)", fontsize=12)
    axs[1].set_ylabel("Bayes factor (log scale)",fontsize=12)
    
    # removing extra white spaces
    # fig.tight_layout()
    
    # showing the figure
    # plt.show()

    # returning the figure
    return fig


# defining a function to compute BFs for differences with chance level for a group of GAT matrices
def bf_testing_gat(
    scores, bf,
    plot_title="Sensor space decoding",
    chance=0.5, x_ticks=np.array([-0.2, 1, -0.2, 1]), xlab=None, ylab=None,
    clim_bf=None, n_timepoints=None
):
    
    # sanity check (these should be the same)
    # print("shape of aggregated scores:", scores.shape, "shape of BFs:", bf.shape)

    # returning an error message if condition is not met
    assert scores.shape == bf.shape, "scores and bf objects should have the same shape"
    
    # defining the plot grid
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=False, sharey=False)
    
    # defining the main title
    # fig.suptitle(plot_title, size=14, weight=800)
    fig.suptitle(plot_title, size=14, weight=800, y=0.8)

    # computing average decoding accuracy
    if len(scores.shape)>2:
        time_gen_avg_scores = np.mean(scores, axis=0)
    else:
        time_gen_avg_scores = scores
    

    # defining color limits for decoding accuracies
    acc_min, acc_max = np.min(time_gen_avg_scores), np.max(time_gen_avg_scores)
    max_range = max(np.abs(acc_min - chance), np.abs(acc_max - chance))
    clim = (chance - max_range, chance + max_range)

    # defining color limits for decoding accuracies
    if clim_bf is None:
        min_bf, max_bf = np.min(bf), np.max(bf)
        # max_range_bf = max(np.abs(min_bf - 1), np.abs(max_bf - 1))
        max_range_log_bf = max(np.abs(np.log(min_bf)), np.log(max_bf))
        clim_bf = (1 / np.exp(max_range_log_bf), np.exp(max_range_log_bf) )

    # plotting the GAT matrix
    im0 = axs[0].imshow(
        time_gen_avg_scores,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=x_ticks,
        vmin=np.min(clim), vmax=np.max(clim)
    )

    # defining the axes titles
    axs[0].set_xlabel(xlab)
    axs[0].set_ylabel(ylab)

    # plotting the stimulus onset
    axs[0].axvline(0, color="k")
    axs[0].axhline(0, color="k")

    # defining the diagonal
    axs[0].axline((1, 1), slope=1, color="k")

    # computing (height_of_image / width_of_image)
    im_ratio = bf.shape[0]/bf.shape[1]

    # defining the colorbar
    # see https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/
    cbar = plt.colorbar(im0, ax=axs[0], fraction=0.046*im_ratio)
    cbar.set_label("Decoding accuracy (AUC)")

    # plotting the BFs for the GAT matrix
    im1 = axs[1].imshow(
        bf,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=x_ticks,
        # vmin=np.min(clim_bf), vmax=np.max(clim_bf)
        # norm=colors.LogNorm(vmin=bf.min(), vmax=bf.max())
        norm=colors.LogNorm(vmin=np.min(clim_bf), vmax=np.max(clim_bf))
    )

    # defining the axes titles
    axs[1].set_xlabel(xlab)
    axs[1].set_ylabel(ylab)

    # plotting the stimulus onset
    axs[1].axvline(0, color="k")
    axs[1].axhline(0, color="k")

    # defining the diagonal
    axs[1].axline((1, 1), slope=1, color="k")

    # defining the colorbar
    cbar = plt.colorbar(im1, ax=axs[1], fraction=0.046*im_ratio)
    cbar.set_label("Bayes factor (log scale)")
    
    # tight layout
    fig.tight_layout()

    # removing extra white spaces
    fig.subplots_adjust(top=0.8, bottom=0.2)

    # returning the figure
    return fig
