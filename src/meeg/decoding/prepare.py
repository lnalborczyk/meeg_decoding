import numpy as np
import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
import scipy
    
# https://copyprogramming.com/howto/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
def trials_avg(epochs_data, labels, N=2, shuffling_or_not=False):

    # initialising the final_lengths array
    final_lengths = []

    # identifying the max values (end of data for each label)
    maxs = (0, np.where(np.diff(labels, prepend=np.nan))[0][1], len(labels))
    
    for i in range(len(maxs)-1): # for each labels' value
        
        # print("current nmax:", maxs[i+1], "and i:", i)
        # print("cutting data from:", maxs[i], "to", maxs[i+1])
        
        # retrieving data with the current label
        current_epochs_data = epochs_data[maxs[i]:maxs[i+1],:,:]

        # computing the number of micro-averaged sets
        nb_of_splits = np.trunc(current_epochs_data.shape[0] / N)
        
        # computing the remainder (left-over trials)
        remainder = current_epochs_data.shape[0] % N

        if remainder > 0:

            final_nb_of_splits = int(nb_of_splits + 1)

        elif remainder == 0:

            final_nb_of_splits = int(nb_of_splits)
        
        # sanity check
        # print("final number of splits for this label:", final_nb_of_splits)
        
        # initialising the result object
        result = []

        if shuffling_or_not:

            np.random.shuffle(current_epochs_data)
            # print("shape after shuffling:", current_epochs_data.shape)

        for j in range(final_nb_of_splits):

            if j > nb_of_splits:

                micro_average = np.mean(current_epochs_data[j*N:,:,:], axis=0)
                result.append(micro_average)
            
            else:
                
                micro_average = np.mean(current_epochs_data[j*N:j*N+N-1,:,:], axis=0)
                result.append(micro_average)

        # storing the length of the current chunk of trials
        result = np.array(result)
        
        # sanity check
        # print("array length:", result.shape)
        
        # appending the final length
        final_lengths.append(result.shape[0])
        
        if i == 0:
    
            new_epochs = result
    
        else:
    
            new_epochs = np.vstack((new_epochs, result))

    # reducing the original labels
    unique_labels = list(set(labels))
    new_labels = [unique_labels[0]] * final_lengths[0] + [unique_labels[1]] * final_lengths[1]

    # returning the new epochs and labels
    return new_epochs, new_labels


def sliding_average(epochs, decim=5):
    '''
    Sliding average time window on the epochs data.
    The time windows are non overlapping. For each time slice, the average of all
    data at those time points is kept (as opposed to keeping all time points as a
    vector).
    For overlapping time windows that keep all time points as a vector, see
    make_temporal_patches()

    epochs : mne.epochs.EpochsFIF or np.array of epochs/stc data
            if np.array, must have dimensions (ntrials x nsource/sensor x ntimes)

    decim : integer to decimate by

    returns
    --------
    downsampled_epoch : np.array of the downsampled epochs

    '''
    if type(epochs) == mne.epochs.EpochsFIF: # convert to np.array if type is mne epochs obj
        epochs_data = epochs.get_data()
    else:
        epochs_data = epochs

    n_times =  epochs_data.shape[2]
    trimmed_times = n_times - n_times%decim # remove extra time points so n_times is divisible by decim
    epochs_data = epochs_data[:,:,:trimmed_times] # drop the additional time points at the end

    target_nTimes = int(trimmed_times / decim) # nb of times for the decimated epochs
    downsampled_epoch = np.zeros([epochs_data.shape[0],epochs_data.shape[1], target_nTimes]) # create output array with zeros

    for i_trial in range(epochs_data.shape[0]): # for each trial
        for i_sensor in range(epochs_data.shape[1]): # for each sensor

            sub = epochs_data[i_trial, i_sensor,:]
            downsampled_epoch[i_trial, i_sensor,:] = np.mean(sub.reshape(-1, decim), axis=1)

    # returning the downsampled epochs
    return downsampled_epoch
    

def PCA_epochs(epochs, n_comp):
    
    '''
    Fit PCA on epochs_fit, and transform epochs_trans using n_comp.
    Prints explained variance when using n_comp.
    '''
    
    mdl_PCA = PCA(n_comp)
    pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
    pca_data = pca.fit_transform(epochs.get_data())

    explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
    print("PCA explained var:%.3f"%explained_var)
    
    return pca_data


def PCA_epochsArray(array, n_comp):
    
    '''
    Fit PCA on epochs_fit, and transform epochs_trans using n_comp.
    Prints explained variance when using n_comp.
    '''
    
    # select the number of dimensions so that explained variance is > 99%
    # pca = UnsupervisedSpatialFilter(PCA(n_components=0.99, svd_solver = 'full'), average=False)
    
    mdl_PCA = PCA(n_comp)
    pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
    pca_data = pca.fit_transform(array)

    explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
    print("PCA explained variance:%.3f"%explained_var)
    
    return pca_data


# defining a function to help with the pre-decoding processing
def prep_data_for_decoding(
    epochs, pca=True, n_components=50,
    moving_average=True, kernel_size=20,
    moving_average_with_decim=False, decim=4,
    trials_averaging=False, ntrials=4, shuffling_or_not=False
):

    # retrieving the MEG signals: n_epochs, n_meg_channels, n_times
    X = epochs.get_data(copy=False)

    # retrieving the labels (i.e., items categories)
    # y = abs((epochs.events[:, 2]-1) // 10)
    y = epochs.events[:, 2]

    # some sanity checks
    print("Original shape of the MEG data:", X.shape)
    print("Length of the labels to be predicted:", len(y))

    if trials_averaging:

        # micro-averaging the MEG data (and the labels accordingly)
        X, y = trials_avg(epochs_data=X, labels=y, N=ntrials, shuffling_or_not=shuffling_or_not)
        print("Shape of the MEG data after trials averaging:", X.shape)
        print("Shape of the labels after trials averaging:", len(y))

    if moving_average:

        # smoothing the MEG signals
        kernel = np.ones(kernel_size) / kernel_size
        X = scipy.ndimage.convolve1d(X, kernel, axis=-1)
        print("Moving average applied with kernel size:", kernel_size)

    if moving_average_with_decim:

        # moving average with some decimation factor
        X = sliding_average(epochs=X, decim=decim)
        print("Moving average applied with decim factor:", decim)
    
    if pca:
        
        # or reduce to n_components dimensions
        X = PCA_epochsArray(array=X, n_comp=n_components)

    # returning the reshaped data and labels
    return X, y

