import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from meeg.decoding import prep_data_for_decoding


def minmax_scale(data):
    """
    Rescale the data to between 0 and 1 using minmax scaling.
    """
    
    min_val = np.min(data)
    max_val = np.max(data)
    rescaled_data = (data - min_val) / (max_val - min_val)

    # returning the rescale data
    return rescaled_data


def pca_through_time(epochs, n_components=10):
    """
    Computing PCA through time (trajectories).

    Parameters
    ----------
    epochs: MNE epochs
        The M/EEG data from which to compute the trajectory.
        Should be a 3-dimensional array/tensor of shape trials x channels x time_steps.

    n_components: int
        Number of PCA components to include.
    """
        
    # computing the PCA through time
    X, y = prep_data_for_decoding(
        epochs=epochs,
        pca=True, n_components=n_components,
        moving_average=False, kernel_size=5,
        moving_average_with_decim=False, decim=4,
        trials_averaging=False, ntrials=4, shuffling_or_not=True
    )
    
    # averaging these data across trials
    x_pca = np.mean(X, axis=0).transpose()
    x_pca_std = np.std(X, axis=0).transpose()
    
    # sanity check
    print("Shape of the MEG data after PCA:", x_pca.shape)

    # returning it
    return x_pca, x_pca_std


def compare_pca_through_time(epochs1, epochs2, n_components=10):
    """
    Computing PCA through time (trajectories) for two epochs in a common space.

    Parameters
    ----------
    epochs1: MNE epochs
        The M/EEG data from which to compute the trajectory.
        Should be a 3-dimensional array/tensor of shape trials x channels x time_steps.

    epochs2: MNE epochs
        The M/EEG data from which to compute the trajectory.
        Should be a 3-dimensional array/tensor of shape trials x channels x time_steps.

    n_components: int
        Number of PCA components to include.
    """
    
    # computing the global pca
    # concatenated_epochs = mne.concatenate_epochs(
    #     epochs_list=[epochs1, epochs2],
    #     add_offset=True, on_mismatch="raise", verbose=None
    #     )
    pca_global = PCA(n_components)
    pca = UnsupervisedSpatialFilter(pca_global, average=False)

    # retrieving the MEG data
    trials1 = epochs1.get_data()
    trials2 = epochs2.get_data()

    # sanity check
    # print("data shapes:", trials1.shape, trials2.shape)

    # getting the minimum number of trials
    nb_trials = min(trials1.shape[0], trials2.shape[0])

    # projecting original data onto a global (common) space
    score1_global = pca.fit_transform(trials1[0:nb_trials, :, :])
    score2_global = pca.fit_transform(trials2[0:nb_trials, :, :])
        
    # averaging these PCA trajectories across trials
    x_pca1 = np.mean(score1_global, axis=0).transpose()
    x_pca_std1 = np.std(score1_global, axis=0).transpose()
    x_pca2 = np.mean(score2_global, axis=0).transpose()
    x_pca_std2 = np.std(score2_global, axis=0).transpose()
    
    # returning it
    return x_pca1, x_pca_std1, x_pca2, x_pca_std2


def stats_trajectories(epochs, n_components=10, standardise=True):
    """
    Computing mean, SD, speed, and curvature of latent trajectories.

    Parameters
    ----------
    epochs: MNE epochs
        The M/EEG data from which to compute the trajectories.
        Should be a 3-dimensional array/tensor of shape trials x channels x time_steps.

    n_components: int
        Number of PCA components to include.

    std_speed_curvature: bool
        Should we standardise (min-max) the speed and curvature?
    """
    
    # reshaping data
    X, y = prep_data_for_decoding(
        epochs=epochs,
        pca=True, n_components=n_components,
        moving_average=False, kernel_size=5,
        moving_average_with_decim=False, decim=4,
        trials_averaging=False, ntrials=4, shuffling_or_not=True
    )
    
    # computing the average trajectory
    pca_mean = np.mean(X, axis=0).transpose()

    # computing the variability (SD) of trajectories across trials
    pca_sd = np.std(X, axis=0).transpose()

    # computing the norm of the SD vector
    sd_norm = np.linalg.norm(pca_sd, axis=1)
    
    # computing the first and second temporal derivatives of the average trajectory
    rp = np.gradient(pca_mean, axis=0)
    rpp = np.gradient(rp, axis=0)

    # computing the curvature when there is less than 4 PCA components
    if pca_mean.shape[1] < 4:
        # solution from https://www.whitman.edu/mathematics/calculus_online/section13.03.html#:~:text=Fortunately%2C%20there%20is%20an%20alternate,′(t)%7C3.
        # computing the cross product rp x rpp
        cross_product = np.cross(rp, rpp)

        # computing the norms
        norm_rp = np.linalg.norm(rp, axis=1)
        norm_cross_product = np.linalg.norm(cross_product, axis=1)

        # computing the curvature k (for 3 components at most)
        curvature = norm_cross_product / (norm_rp**3)

    # else, if there is more than 3 PCA components    
    else:

        # computing the norms
        norm_rp = np.linalg.norm(rp, axis=1)

        # computing the curvature k (for more than 3 components)
        # here, we use the approximation |r' x r''| / |r'|^3
        # for higher dimensions, we can compute the norm of the projection of rpp onto the normal plane of rp
        projection = rpp - (np.sum(rp * rpp, axis=1) / norm_rp**2)[:, np.newaxis] * rp
        norm_projection = np.linalg.norm(projection, axis=1)
        curvature = norm_projection / norm_rp**3
        

    # standardise stats between 0 (min) and 1 (max)
    if standardise:
        pca_sd = minmax_scale(sd_norm)
        norm_rp = minmax_scale(norm_rp)
        curvature = minmax_scale(curvature)

    # returning the trajectories and stats
    return pca_mean, pca_sd, norm_rp, curvature
