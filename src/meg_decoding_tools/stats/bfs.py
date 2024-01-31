import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import multiprocessing as mp
import seaborn as sns
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
bf_package=importr("BayesFactor")


def compute_bf(data, i=None, j=None):

    #if i is None:

        # extracting current cell and converting it to a dataframe
    #    df = pd.DataFrame(data)

    #else:
        
        # extracting current cell and converting it to a dataframe
    #    df = pd.DataFrame(data[:, i, j])

    # extracting current cell and converting it to a dataframe
    df = pd.DataFrame(data)
    
    # converting it to R object
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(df)
        
    # computing the BF
    # results=bf_package.ttestBF(x=r_data[0], mu=0, rscale="medium", nullInterval=[float("-inf"), 0])
    results=bf_package.ttestBF(x=r_data[0], mu=0, rscale="medium", nullInterval=[0.5, float("inf")])
        
    # storing the BF in favour of accuracy above chance versus point null
    bf = np.asarray(r["as.vector"](results))[0]

    # storing the BF in favour of the accuracy being above chance versus below chance
    # bf = np.asarray(r["as.vector"](results))[0] / np.asarray(r["as.vector"](results))[1]

    # updating the progress bar
    # pbar.update(1)
    
    # returning the bf
    return bf


# this helper function is needed because map() can only be used for functions
# that take a single argument (see http://stackoverflow.com/q/5442910/1461210)
def compute_bf_parallel(args):
    
    return compute_bf(*args)


# defining a function to compute BFs for differences with chance levels for a group of decoding accuracies over time
def bf_testing_time_decod(
    participants=None, participants_type=None, contrast=None, npy_folder=None,
    plot_title="Sensor space decoding (3l2s vs. 7l2s) in auditory blocks for control participants (N=3)",
    chance=0.5
):

    # filtering participants
    if participants_type == "all":
        
        participants = participants
        
    elif participants_type == "tts":
    
        participants = list(filter(lambda k: "_t" in k, participants))
    
    elif participants_type == "control":
    
        participants = list(filter(lambda k: "_c" in k, participants))
    
    # sanity check
    print("participants:", participants)

    # defining the file name
    fname = npy_folder + contrast + ".npy"
    
    # initialising an empty list
    scores_arr = []
    
    for ppt in participants:
    
        decoding_results_temp = np.load(ppt + fname)
        scores_arr.append(np.mean(decoding_results_temp, axis=0))
    
    
    # converting back to numpy array
    scores = np.vstack(scores_arr)
    
    # sanity check
    print("shape of aggregated scores:", scores.shape)

    # converting scores to a dataframe [this should be participants x timepoints accuracy matrix]
    df = pd.DataFrame(scores)
    
    # loop over timepoints, make decoding accuracy into effect size and convert to an r object
    n_timepoints = df.shape[1]
    df_norm = pd.DataFrame(np.empty_like(df))
    
    for t in range(n_timepoints):
      df_norm[t]=[(i - chance) for i in df[t]]
     
    with localconverter(ro.default_converter + pandas2ri.converter):
      r_data = ro.conversion.py2rpy(df_norm)
    
    # initialising an empty array to store the BFs
    bf = []
    
    # looping over timepoints
    for t in range(n_timepoints):
        
        results=bf_package.ttestBF(x=r_data[t], mu=0, rscale="medium", nullInterval=[0.5, float("inf")])
        bf.append(np.asarray(r["as.vector"](results))[0])

    # defining the grid
    fig, axs = plt.subplots(2, figsize=(12, 6))
    
    # defining the main title
    fig.suptitle(plot_title, size=14, weight=800)
    set_title

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
    markerline, stemlines, baseline = axs[1].stem(x, y,bottom=1,linefmt='k', markerfmt=None, basefmt=None)
    
    markerline.set_markerfacecolor('w')
    markerline.set_markeredgecolor('w')
    baseline.set_color('k')
    stemlines.set_linewidth(0.5)
    
    cols_idx = [np.argmin(np.abs(val_col_map-i)) for i in y]  
    
    # plotting the stimulus onset
    axs[1].axvline(x=0, color="k", ls="-")
    [axs[1].plot(x[i],y[i],color=bf_cols[cols_idx[i]],marker='.',markersize=8,lw=0,markeredgecolor=None) for i in range(len(cols_idx))]
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

    # returning the figure and BFs
    return fig, bf


# defining a function to compute BFs for differences with chance level for a group of GAT matrices
def bf_testing_gat(
    bf=None, participants=None, participants_type=None, contrast=None, npy_folder=None,
    plot_title="Sensor space decoding (3l2s vs. 7l2s) in auditory blocks for control participants (N=3)",
    chance=0.5, x_ticks=np.array([-0.2, 1, -0.2, 1]), xlab=None, ylab=None,
    clim_bf=None, n_timepoints=None, ncores=1
):

    # filtering participants
    if participants_type == "all":
        
        participants = participants
        
    elif participants_type == "tts":
    
        participants = list(filter(lambda k: "_t" in k, participants))
    
    elif participants_type == "control":
    
        participants = list(filter(lambda k: "_c" in k, participants))
    
    # sanity check
    print("Participants:", participants)

    # defining the file name
    fname = npy_folder + contrast + ".npy"
    
    # initialising an empty list
    scores_arr = []
    
    for ppt in participants:
    
        decoding_results_temp = np.load(ppt + fname)

        # if the results contain more than 2 dimensions (e.g., multiple CV folds), computing the average decoding accuracy
        if len(decoding_results_temp.shape)>2:
            decoding_results_temp = np.mean(decoding_results_temp, axis=0)
        
        scores_arr.append(decoding_results_temp)
    
    
    # converting back to numpy array
    scores = np.stack(scores_arr)
    
    # sanity check
    print("Shape of aggregated scores:", scores.shape)

    if bf is None:
        
        # retrieving the number of timepoints
        if n_timepoints is None:
            n_timepoints = scores.shape[2]
        
        # initialising an empty 2D array to store the BFs
        bf = np.zeros((n_timepoints, n_timepoints))
    
        # if sequential
        if ncores==1:
    
            # sanity check
            print("Sequential mode = -_-'")
            
            # looping over timepoints, converting decoding accuracy into effect size and computing the BF10
            for i in range(n_timepoints):
        
                # printing progress
                print("Processing row number", i+1, "out of", n_timepoints, "rows.")
                
                for j in range(n_timepoints):
    
                    # computing and storing the BF
                    # bf[i, j] = compute_bf(scores-chance, i, j)
                    bf[i, j] = compute_bf(data=scores[:, i, j]-chance)
            
    
        elif ncores > 1: # or if parallel
    
            # sanity check
            print("Parallel mode = \_ô_/")
    
            # initialising the progress bar
            # pbar = tqdm(total=n_timepoints**2)
    
            # creating a process pool that uses ncores cpus
            pool = mp.Pool(processes=ncores)
            
            # computing and storing the BF
            # https://stackoverflow.com/questions/29857498/how-to-apply-a-function-to-a-2d-numpy-array-with-multiprocessing
            # bf = np.array(pool.map(multiple_args, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints)
            # bf = np.array(pool.map(multiple_args, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints)
    
            # bf = np.array(pool.apply_async(multiple_args, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)), callback=update)).reshape(n_timepoints, n_timepoints)
    
            # for _ in tqdm(np.array(pool.imap_unordered(multiple_args, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints), total=n_timepoints**2):
            #    pass
    
            # bf = tqdm(np.array(pool.map(multiple_args, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints), total=n_timepoints**2)
    
            # this makes the kernel dies (too heavy on memory)...
            # bf = np.array(pool.map(compute_bf_parallel, ((scores-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)), chunksize=None)).reshape(n_timepoints, n_timepoints)
    
            bf = np.array(pool.map(compute_bf_parallel, ((scores[:, i, j]-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints)
    
            # sanity check
            print("Shape of bf object:", bf.shape)
            
            # closing the process pool
            pool.close()
    
            # waiting for all issued tasks to complete
            pool.join()
            
            # closing the progress bar
            # pbar.close()
        
            # converting the BFs to log-BFs
            # bf = np.log(bf)
    
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

    # returning the figure and BFs
    return fig, bf

