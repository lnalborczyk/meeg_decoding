import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from adjustText import adjust_text
from meeg.latent import stats_trajectories 


def plot_neural_trajectories_2d(
    x_pca, x_pca_std=None, fs=1000, cmap="magma", plot_title="Neural trajectories"
    ):

    # assigning colours to timepoints
    timepoints = np.arange(x_pca.shape[0]) / (1 / fs)
    colormap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )

    # plotting (normalised) variability as dot size/transparency
    if x_pca_std is None:
        
        scatter_alpha = 1

    else:

        x_pca_std_dims_average = np.mean(x_pca_std, axis=1)
        x_norm = (x_pca_std_dims_average - np.min(x_pca_std_dims_average)) / (np.max(x_pca_std_dims_average) - np.min(x_pca_std_dims_average))
        scatter_alpha = 1 - x_norm

    # plotting time
    fig = plt.figure(figsize = (12, 9) )
    scatter = plt.scatter(x_pca[:,0], x_pca[:,1], c = timepoints, cmap = colormap, norm = norm, alpha = scatter_alpha, s = 20 * scatter_alpha)

    # defining axes labels etc
    plt.colorbar(scatter, label = "Time (s)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(plot_title)
    
    # polishing the layout
    plt.tight_layout()

    # showing the plot
    plt.show()

    # returning the figure
    return fig


def plot_neural_trajectories_2d_with_stats(epochs, n_components=10, cmap="magma"):

    # computing stats
    pca_mean, pca_sd, speed, curvature = stats_trajectories(epochs=epochs, n_components=n_components)

    # retrieving sampling frequency
    fs = epochs.info["sfreq"]

    # retrieving x-axis ticks
    x_ticks = epochs.times

    # defining the grid
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))

    # defining the colorbar
    colormap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin = x_ticks.min(), vmax = x_ticks.max() )
    
    # making the plot
    axs[0].plot(pca_mean[:,0], pca_mean[:,1], linestyle = "--", linewidth = 1, color = "black")
    im0 = axs[0].scatter(pca_mean[:,0], pca_mean[:,1], c = x_ticks, cmap = colormap, norm = norm, alpha = 1, s = 15)
    axs[0].set_xlabel("PC1", fontsize=10)
    axs[0].set_ylabel("PC2", fontsize=10)
    cbar = plt.colorbar(im0, ax=axs[0], fraction=0.046)
    cbar.set_label("Time (s)")

    # baseline starts
    baseline_onset = 0
    # print(baseline_onset)
    axs[0].plot(pca_mean[baseline_onset, 0], pca_mean[baseline_onset, 1], marker = "o", c = "black", markersize=10)

    # stimulus onset
    stim_onset = int(0.2 / (1 / fs) )
    # print(stim_onset)
    axs[0].plot(pca_mean[stim_onset, 0], pca_mean[stim_onset, 1], marker = "o", c = "green", markersize=10)

    # maximum speed
    speed_max = np.argmax(speed)
    axs[0].plot(pca_mean[speed_max, 0], pca_mean[speed_max, 1], marker = "o", c = "orange", markersize=10)

    # maximum curvature
    curv_max = np.argmax(curvature)
    axs[0].plot(pca_mean[curv_max, 0], pca_mean[curv_max, 1], marker = "o", c = "blue", markersize=10)

    # plotting the timecourse of speed and curvature
    axs[1].plot(x_ticks, speed, label="speed")
    axs[1].plot(x_ticks, curvature, label="curvature")
    axs[1].plot(x_ticks, pca_sd, label="traj_sd")
    axs[1].set_xlabel("Time (s)", fontsize=10)
    axs[1].set_ylabel("Rescaled Speed/Curvature/SD", fontsize=10)
    # axs[1].set_ylabel("Rescaled Speed/Curvature", fontsize=10)
    plt.legend()
    plt.grid(True)

    # polishing the layout
    plt.tight_layout()

    # showing the plot
    plt.show()

    # returning the figure
    return fig


def plot_neural_trajectories_3d(
        x_pca, x_pca_std=None, fs=1000, cmap="magma", plot_title="Neural trajectories"
        ):

    # assigning colours to timepoints
    timepoints = np.arange(x_pca.shape[0]) / (1 / fs)
    colormap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )

    # plotting (normalised) variability as dot transparency
    if x_pca_std is None:
        
        scatter_alpha = 1

    else:

        x_pca_std_dims_average = np.mean(x_pca_std, axis=1)
        x_norm = (x_pca_std_dims_average - np.min(x_pca_std_dims_average)) / (np.max(x_pca_std_dims_average) - np.min(x_pca_std_dims_average))
        scatter_alpha = 1-x_norm
        
    # plotting time
    fig = plt.figure(figsize = (12, 9) )
    ax = plt.axes(projection = "3d")
    
    sc = ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c = timepoints, cmap = colormap, s = 30, alpha = scatter_alpha)
    
    ax.view_init(20, 20, 0)
    plt.colorbar(sc, label = "Time (s)")
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(plot_title)
    
    # polishing the layout
    plt.tight_layout()

    # showing the plot
    plt.show()

    # returning the figure
    return fig


def compare_neural_trajectories_2d(
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None,
    fs=1000,
    plot_title="Neural trajectories"
    ):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca1.shape[0]) / (1 / fs)
    colormap1 = cm.get_cmap("Blues")
    colormap2 = cm.get_cmap("Oranges")
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )

    # plotting (normalised) variability as dot transparency
    if x_pca_std1 is None:
        
        scatter_alpha1 = 1
        scatter_alpha2 = 1

    else:

        x_pca_std_dims_average1 = np.mean(x_pca_std1, axis=1)
        x_norm1 = (x_pca_std_dims_average1 - np.min(x_pca_std_dims_average1)) / (np.max(x_pca_std_dims_average1) - np.min(x_pca_std_dims_average1))
        scatter_alpha1 = 1 - x_norm1
        x_pca_std_dims_average2 = np.mean(x_pca_std2, axis=1)
        x_norm2 = (x_pca_std_dims_average2 - np.min(x_pca_std_dims_average2)) / (np.max(x_pca_std_dims_average2) - np.min(x_pca_std_dims_average2))
        scatter_alpha2 = 1 - x_norm2
        
    # plotting time
    fig = plt.figure(figsize = (16, 9) )
    
    scatter1 = plt.scatter(
        x_pca1[:,0], x_pca1[:,1], c = timepoints, cmap = colormap1, norm = norm,
        # alpha = scatter_alpha1,
        s = 20 * scatter_alpha1
    )

    scatter2 = plt.scatter(
        x_pca2[:,0], x_pca2[:,1], c = timepoints, cmap = colormap2, norm = norm,
        # alpha = scatter_alpha2,
        s = 20 * scatter_alpha2
    )

    # defining axes labels etc
    plt.colorbar(scatter1, label = "Time (s)")
    plt.colorbar(scatter2, label = "Time (s)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(plot_title)
    
    # polishing the layout
    plt.tight_layout()

    # showing the plot
    plt.show()

    # returning the figure
    return fig


def compare_neural_trajectories_3d(
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None,
    fs=1000,
    plot_title="Neural trajectories"
    ):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca1.shape[0]) / (1 / fs)
    # colormap = cm.get_cmap(cmap)
    colormap1 = cm.get_cmap("Blues")
    colormap2 = cm.get_cmap("Oranges")
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )

    # plotting (normalised) variability as dot transparency
    if x_pca_std1 is None:
        
        scatter_size1 = 20
        scatter_size2 = 20

    else:

        x_pca_std_dims_average1 = np.mean(x_pca_std1, axis=1)
        x_norm1 = (x_pca_std_dims_average1 - np.min(x_pca_std_dims_average1)) / (np.max(x_pca_std_dims_average1) - np.min(x_pca_std_dims_average1))
        scatter_size1 = 1-x_norm1
        x_pca_std_dims_average2 = np.mean(x_pca_std2, axis=1)
        x_norm2 = (x_pca_std_dims_average2 - np.min(x_pca_std_dims_average2)) / (np.max(x_pca_std_dims_average2) - np.min(x_pca_std_dims_average2))
        scatter_size2 = 1-x_norm2
        
    # plotting time
    fig = plt.figure(figsize = (12, 9) )
    ax = plt.axes(projection = "3d")
    
    sc1 = ax.scatter(x_pca1[:, 0], x_pca1[:, 1], x_pca1[:, 2], c = timepoints, cmap = colormap1, s = 20 * scatter_size1)
    sc2 = ax.scatter(x_pca2[:, 0], x_pca2[:, 1], x_pca2[:, 2], c = timepoints, cmap = colormap2, s = 20 * scatter_size2)
    
    ax.view_init(22.5, 45, 0)

    # defining axes labels etc
    plt.colorbar(sc1, label = "Time (s)")
    plt.colorbar(sc2, label = "Time (s)")
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(plot_title)

    # polishing the layout
    plt.tight_layout()

    # showing the plot
    plt.show()

    return fig


def compare_neural_trajectories_3d_animated(
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None,
    fs=1000, time_interval=100,
    plot_title="Neural trajectories"
    ):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca1.shape[0]) / (1 / fs)
    colormap1 = cm.get_cmap("Blues")
    colormap2 = cm.get_cmap("Oranges")
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )
    
    # plotting (normalised) variability as dot transparency
    if x_pca_std1 is None:
        
        scatter_size1 = 20
        scatter_size2 = 20

    else:

        x_pca_std_dims_average1 = np.mean(x_pca_std1, axis=1)
        x_norm1 = (x_pca_std_dims_average1 - np.min(x_pca_std_dims_average1)) / (np.max(x_pca_std_dims_average1) - np.min(x_pca_std_dims_average1))
        scatter_size1 = 1-x_norm1
        x_pca_std_dims_average2 = np.mean(x_pca_std2, axis=1)
        x_norm2 = (x_pca_std_dims_average2 - np.min(x_pca_std_dims_average2)) / (np.max(x_pca_std_dims_average2) - np.min(x_pca_std_dims_average2))
        scatter_size2 = 1-x_norm2

    
    # defining the plot updating function
    def update(num, x_pca1, x_pca2, line1, line2, sc1, sc2):
        
        # line1.set_data(matching_data[:2, :num])
        # line1.set_3d_properties(matching_data[2, :num])
        # line2.set_data(non_matching_data[:2, :num])
        # line2.set_3d_properties(non_matching_data[2, :num])
        
        # sc1.set_data(x_pca1[:2, num])
        # sc1.set_3d_properties(x_pca1[2, num])
        # sc2.set_data(x_pca2[:2, num])
        # sc2.set_3d_properties(x_pca2[2, num])

        line1.set_data(np.transpose(x_pca1)[:2, :num])
        line1.set_3d_properties(np.transpose(x_pca1)[2, :num])
        line2.set_data(np.transpose(x_pca2)[:2, :num])
        line2.set_3d_properties(np.transpose(x_pca2)[2, :num])

        # sc1.set_data(np.transpose(x_pca1)[:2, num])
        # sc1.set_3d_properties(np.transpose(x_pca1)[2, num])
        # sc2.set_data(np.transpose(x_pca2)[:2, num])
        # sc2.set_3d_properties(np.transpose(x_pca2)[2, num])

        # line1.set_data(x_pca1[:num, :2])
        # line1.set_3d_properties(x_pca1[:num, 2])
        # line2.set_data(x_pca2[:num, :2])
        # line2.set_3d_properties(x_pca2[:num, 2])

        sc1.set_data(x_pca1[num, :2])
        sc1.set_3d_properties(x_pca1[num, 2])
        sc1.set(ms = 20 * scatter_size1[num])
        sc2.set_data(x_pca2[num, :2])
        sc2.set_3d_properties(x_pca2[num, 2])
        sc2.set(ms = 20 * scatter_size2[num])

        # defining the figure's title
        ax.set_title("\n\n" + plot_title + "\n\nTime: " + str(format(num/fs, ".2f")) + "s", fontsize=16)


    # plotting time
    # initialising the 3D plot
    fig = plt.figure(figsize = (12, 9) )
    ax = plt.axes(projection = "3d")

    line1, = ax.plot(x_pca1[0:1, 0], x_pca1[0:1, 1], x_pca1[0:1, 2], lw = 1, c = "#1f77b4", alpha = 0.5)
    line2, = ax.plot(x_pca2[0:1, 0], x_pca2[0:1, 1], x_pca2[0:1, 2], lw = 1, c = "#ff7f0e", alpha = 0.5)
    # sc1 = ax.scatter(x_pca1[0, 0], x_pca1[0, 1], x_pca1[0, 2], c = timepoints[0], cmap = colormap1, s = 20 * scatter_size1[0])
    # sc2 = ax.scatter(x_pca2[0, 0], x_pca2[0, 1], x_pca2[0, 2], c = timepoints[0], cmap = colormap2, s = 20 * scatter_size2[0])
    # sc1 = plt.plot(x_pca1[0, 0], x_pca1[0, 1], x_pca1[0, 2], c = "b", marker = "o", cmap = colormap1, s = 20 * scatter_size1[0])[0]
    # sc2 = plt.plot(x_pca2[0, 0], x_pca2[0, 1], x_pca2[0, 2], c = "r", marker = "o", cmap = colormap2, s = 20 * scatter_size2[0])[0]
    sc1 = plt.plot(x_pca1[0, 0], x_pca1[0, 1], x_pca1[0, 2], c = "#1f77b4", marker = "o")[0]
    sc2 = plt.plot(x_pca2[0, 0], x_pca2[0, 1], x_pca2[0, 2], c = "#ff7f0e", marker = "o")[0]

    # plotting crucial timepoints
    time_point = 0
    plt.plot(x_pca1[time_point, 0], x_pca1[time_point, 1], x_pca1[time_point, 2], marker = "o", c = "#1f77b4")
    plt.plot(x_pca2[time_point, 0], x_pca2[time_point, 1], x_pca2[time_point, 2], marker = "o", c = "#ff7f0e")

    # stimulus onset
    time_point = int(0.2 / (1 / fs) )
    plt.plot(x_pca1[time_point, 0], x_pca1[time_point, 1], x_pca1[time_point, 2], marker = "o", c = "green")
    plt.plot(x_pca2[time_point, 0], x_pca2[time_point, 1], x_pca2[time_point, 2], marker = "o", c = "green")

    # stimulus offset
    time_point = int(0.4 / (1 / fs) )
    plt.plot(x_pca1[time_point, 0], x_pca1[time_point, 1], x_pca1[time_point, 2], marker = "o", c = "red")
    plt.plot(x_pca2[time_point, 0], x_pca2[time_point, 1], x_pca2[time_point, 2], marker = "o", c = "red")
    
    ax.view_init(22.5, 45, 0)

    # animating the plot
    N = x_pca1.shape[0]

    # setting the axes properties
    # ax.set(xlim3d = (-2*1e-11, 2*1e-11), xlabel = "PC1")
    # ax.set(ylim3d = (-2*1e-11, 2*1e-11), ylabel = "PC2")
    # ax.set(zlim3d = (-2*1e-11, 2*1e-11), zlabel = "PC3")
    ax.set(xlim3d = (-1.5*1e-11, 1.5*1e-11), xlabel = "PC1")
    ax.set(ylim3d = (-1.5*1e-11, 1.5*1e-11), ylabel = "PC2")
    ax.set(zlim3d = (-1.5*1e-11, 1.5*1e-11), zlabel = "PC3")

    anim = animation.FuncAnimation(
        fig, update, N, fargs = (x_pca1, x_pca2, line1, line2, sc1, sc2),
        interval = time_interval, blit = False
    )

    # showing the plot
    plt.show()

    # returning the animated plot
    return anim


def plot_embeddings(points, points_color, labels, plot_title):
    
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white", constrained_layout=True)
    fig.suptitle(plot_title, size=16)
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=100, alpha=0.8)
    texts = []
    
    for x, y, s in zip(x, y, labels):
        texts.append(plt.text(x, y, s))
    
    adjust_text(texts, only_move={"points":"y", "texts":"y"}, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))

