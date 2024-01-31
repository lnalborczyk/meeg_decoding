
import numpy as np
import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA


def pca_through_time(epochs, n_components=10):
    
    # reshaping data
    X, y = prep_data_for_decoding(
        epochs=epochs,
        pca=True, n_components=n_components,
        moving_average=False, kernel_size=5,
        moving_average_with_decim=False, decim=4,
        trials_averaging=False, ntrials=4, shuffling_or_not=True
    )

    # trying out t-SNE instead?
    # !pip install threadpoolctl==3.1.0
    # tsne = TSNE(n_components = 2, perplexity = 100)
    # x_tsne = tsne.fit_transform(np.transpose(X) )
    # print("Shape of tSNE object:", x_tsne.shape)
    
    # averaging these data across trials
    x_pca = np.mean(X, axis=0).transpose()
    x_pca_std = np.std(X, axis=0).transpose()
    
    # sanity check
    print("Shape of the MEG data after PCA:", x_pca.shape)

    # returning it
    return x_pca, x_pca_std


def compare_pca_through_time(epochs1, epochs2, n_components=10):
    
    # computing the global pca
    concatenated_epochs=mne.concatenate_epochs(
        epochs_list=[epochs1, epochs2],
        add_offset=True, on_mismatch="raise", verbose=None
        )

    pca_global = PCA(n_components)
    pca = UnsupervisedSpatialFilter(pca_global, average=False)
    pca_data = pca.fit_transform(concatenated_epochs.get_data())

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

