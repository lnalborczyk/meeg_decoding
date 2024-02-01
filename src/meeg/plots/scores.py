import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


# defining a function to plot the decoding results
def plotting_decoding_scores(
    decoding_scores, x_ticks, plot_title, end_stim=0.2, plotting_theme="ticks",
    significant_clusters=None
):

    # reshaping these results in a pandas dataframe
    results_df = pd.DataFrame(decoding_scores.transpose())
    # results_df["time"] = epochs.times
    results_df["time"] = x_ticks
    results_df = pd.melt(results_df, id_vars="time", var_name="fold", value_name="score")
    results_df.head()
    
    # plotting theme
    sns.set_theme(style=plotting_theme)
    
    # plotting using sns
    fig, ax = plt.subplots(1, figsize=[12, 6])
    
    # plotting the chance level
    ax.axhline(y=0.5, color="k", ls="--", label="Chance level")
    
    # plotting the stimulus onset
    ax.axvline(x=0, color="k", ls="-")
    
    # plotting the stimulus duration (in visual blocks)
    ax.axvspan(0, end_stim, alpha=0.1, color="black")
    
    # plotting the average decoding accuracy (over folds) with 95% confidence interval
    sns.lineplot(
        data=results_df, x="time", y="score", ax=ax, lw=2,
        estimator="mean", errorbar=("ci", 95), n_boot=1000, label="Average accuracy"
    )
    
    # plotting timepoint significantly better than chance
    if significant_clusters is not None:
        
        for i in range(len(significant_clusters)):
            # ax.plot(epochs.times[significant_clusters_x[i]], significant_clusters_y, marker="o", color="b", markersize=5)
            ax.plot(epochs.times[significant_clusters[i]], np.min(np.mean(decoding_scores, axis=0)), marker="o", color="b", markersize=5)
    
    # filling accuracy above 0.5
    # plt.fill_between(x=results_df["time"], y1=0.5, y2=results_df["score"], alpha=0.3, where=results_df["score"]>0.5, interpolate=True)
    
    # specifying axis labels
    ax.set_title(plot_title, size=14, weight=800)
    ax.set_xlabel("Time (s)", size=12)
    ax.set_ylabel("Decoding accuracy (AUC)", size=12)
    
    # adding a legend
    plt.legend(loc="upper right")

    # polishing the layout
    plt.tight_layout()


# defining a function to plot the generalisation across time
def plotting_gat(
    scores, x_ticks=None, epochs=None, plot_title="No title", xlab=None, ylab=None,
    chance=0.5, clim=None, colorbar=True, colorlab=None, diagonal=True
):
        
    # computing average decoding accuracy
    if len(scores.shape)>2:
        time_gen_avg_scores = np.mean(scores, axis=0)
    else:
        time_gen_avg_scores = scores
    
    # plotting the full (temporal generalisation) matrix
    fig, ax = plt.subplots(1, figsize=[12, 9])

    if clim is None:
        acc_min, acc_max = np.min(scores), np.max(scores)
        print("scores min and max:", acc_min, acc_max)
        max_range = max(np.abs(acc_min - chance), np.abs(acc_max - chance))
        clim = (chance - max_range, chance + max_range)

    if x_ticks is None:
        x_ticks=epochs.times[[0, -1, 0, -1]]
        
    im = ax.imshow(
        time_gen_avg_scores,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        #extent=epochs.times[[0, -1, 0, -1]],
        extent=x_ticks,
        vmin=np.min(clim), vmax=np.max(clim)
    )
    
    if xlab is not None:
        ax.set_xlabel(xlab)
    else:
        ax.set_xlabel("Testing time (s)")
    
    if ylab is not None:
        ax.set_ylabel(ylab)
    else:
        ax.set_ylabel("Training time (in seconds)")

    ax.set_title(plot_title, fontweight="bold")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

    if diagonal:

        ax.axline((1, 1), slope=1, color="red")
    
    if colorbar:
        
        cbar = plt.colorbar(im, ax=ax)

        if colorlab is not None:
            cbar.set_label(colorlab)
        else:
            cbar.set_label("Decoding accuracy (AUC)")

    # polishing the layout
    plt.tight_layout()

