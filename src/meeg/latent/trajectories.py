import numpy as np
import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
from meeg.decoding import prep_data_for_decoding


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
    concatenated_epochs=mne.concatenate_epochs(
        epochs_list=[epochs1, epochs2],
        add_offset=True, on_mismatch="raise", verbose=None
        )

    pca_global = PCA(n_components)
    pca = UnsupervisedSpatialFilter(pca_global, average=False)

    # projecting original data onto a global (common) space
    trials1 = epochs1.get_data()
    trials2 = epochs2.get_data()
    score1_global = pca.fit_transform(trials1[0:794, :, :])
    score2_global = pca.fit_transform(trials2[0:794, :, :])
        
    # averaging these PCA trajectories across trials
    x_pca1 = np.mean(score1_global, axis=0).transpose()
    x_pca_std1 = np.std(score1_global, axis=0).transpose()
    x_pca2 = np.mean(score2_global, axis=0).transpose()
    x_pca_std2 = np.std(score2_global, axis=0).transpose()
    
    # returning it
    return x_pca1, x_pca_std1, x_pca2, x_pca_std2

def stats_trajectories(epochs, n_components=10):
    """
    Computing mean, std, speed, and curvature of latent trajectory.

    Parameters
    ----------
    epochs: MNE epochs
        The M/EEG data from which to compute the trajectory.
        Should be a 3-dimensional array/tensor of shape trials x channels x time_steps.

    n_components: int
        Number of PCA components to include.
    """
    
    # reshaping data
    X, y = prep_data_for_decoding(
        epochs=epochs,
        pca=True, n_components=n_components,
        moving_average=False, kernel_size=5,
        moving_average_with_decim=False, decim=4,
        trials_averaging=False, ntrials=4, shuffling_or_not=True
    )
    
    # averaging these data across trials
    pca_mean = np.mean(X, axis=0).transpose()
    pca_std = np.std(X, axis=0).transpose()

    # computing the derivatives of each dimension (i.e., velocity)
    x_t = np.gradient(pca_mean[:, 0])
    y_t = np.gradient(pca_mean[:, 1])
    z_t = np.gradient(pca_mean[:, 2])

    # retrieving velocity for each dimension
    velocity = np.array([[x_t[i], y_t[i], z_t[i]] for i in range(x_t.size)])

    # computing speed (modulus of the velocity)
    speed = np.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    # computing the tangent, we will perform some transformation which will ensure that the size of the speed and velocity is the same.
    # also, we need to be able to divide the vector-valued velocity function to the scalar speed array.
    # https://www.delftstack.com/howto/numpy/curvature-formula-numpy/
    # tangent = np.array([1/speed] * 3).transpose() * velocity

    # computing the curvature
    # ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)
    zz_t = np.gradient(z_t)
    curvature= np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5

    # computing the acceleration
    # https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
    # t_component = np.array([ss_t] * 3).transpose()
    # n_component = np.array([curvature * speed * speed] * 3).transpose()
    # acceleration = t_component * tangent + n_component * normal

    # returning it
    return pca_mean, pca_std, speed, curvature

