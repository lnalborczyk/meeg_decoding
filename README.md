# meg_decoding_tools (work in progress)

Open source tools for M/EEG multivariate pattern analyses (aka decoding) and state-space analyses, based on [MNE-Python](https://mne.tools/stable/index.html).

# Installation

Install directly from Github with:

```
python3 -m pip install git+https://github.com/lnalborczyk/meg_decoding_tools
```

Or clone this repository and install locally with:

```
git clone https://github.com/lnalborczyk/meg_decoding_tools
cd meg_decoding_tools
python3 -m pip install .
```

# Usage

Functions from this package assume that you have some M/EEG data that is properly filtered, resampled, and epoched using [MNE-Python](https://mne.tools/stable/index.html) (i.e., it does not cover preprocessing).

## Decoding

### Decoding through time

```
# importing mne and sub-packages from meg_decoding_tools
import mne
from meeg.decoding.decode import time_decode
from meeg.decoding.prepare import prep_data_for_decoding

# for decoding, we'll keep only two categories and concatenate those
decoding_epochs = mne.concatenate_epochs(
    epochs_list=[some_epochs, some_other_epochs],
    add_offset=True, on_mismatch="raise", verbose=None
    )

# preparing MNE epochs and labels for decoding
X, y = prep_data_for_decoding(
    epochs=decoding_epochs,
    pca=False, n_components=60,
    moving_average=True, kernel_size=5,
    trials_averaging=False, ntrials=2, shuffling_or_not=True
)

# decoding time!
scores, time_decod = time_decode(meg_data=X, labels=y)
```

### Cross-temporal and cross-task generalisation

```
from meeg.decoding.decode import cross_time_cond_gen

# decoding time!
time_gen_scores, decision_values, y_predicted_probs = cross_time_cond_gen(X_train, X_test, y_train, y_test)
```

### Group-level hypothesis testing (based on default Bayes factors)

For decoding through time.

```
import glob
from meeg.stats.decode bfs import bf_testing_time_decod

# listing all relevant npy files (i.e., individual-level decoding accuracies through time)
npy_files = glob.glob("some_directory/+"*.npy")

# initialising an empty list to store the arrays
scores_arr = []

for i in npy_files:

    decoding_results_temp = np.load(i)
    scores_arr.append(np.mean(decoding_results_temp, axis=0))


# converting back to numpy array
scores = np.vstack(scores_arr)

# sanity check (should be of shape n_participants x n_time_steps)
print("shape of aggregated scores:", scores.shape)

# computing the BFs for each time step
bfs = bf_testing_time_decod(scores=scores, ncores=4)
```

Or for cross-temporal and/or cross-condition decoding generalisation.

```
from meeg.stats.decode bfs import bf_testing_gat

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

# computing the BFs for each cell of the GAT matrices
bfs = bf_testing_gat(scores=scores, ncores=4)
```

## Visualisation

### Decoding accuracy through time

```
from meeg.plots.scores import plotting_decoding_scores

# plotting the decoding accuracy over time
plotting_decoding_scores(
    decoding_scores=scores,
    x_ticks=decoding_epochs.times,
    end_stim=0.2,
    plot_title="Sensor space decoding"
)
```

### Decoding accuracy through time with BFs

```
from meeg.plots.bfs import bf_testing_time_decod

# plotting the BFs for each time step
bf_testing_time_decod(scores, bf, plot_title="Sensor space decoding")
```

### Decoding generalisation

```
from meeg.plots.scores import plotting_gat

plotting_gat(
    scores=time_gen_scores,
    epochs=decoding_epochs,
    plot_title="Temporal generalisation matrix"
)
```

### Decoding generalisation with BFs

```
from meeg.plots.bfs import bf_testing_gat

# plotting the BFs for each cell of the GAT matrix
bf_testing_time_decod(scores, bf, plot_title="Sensor space decoding")
```

## State-space trajectories (latent module)

### Neural trajectories

```
from meeg.latent.trajectories import compare_pca_through_time

# compute average neural trajectories (and across-trial SD) in a common PCA space
epochs1_pca, epochs1_pca_std, epochs2_pca, epochs2_pca_std = meg.compare_pca_through_time(epochs1, epochs2, n_components=10)
```

### Dynamical similarity analysis

```
from meeg.latent.dsa import dsa

# commuting dynamical similarity analysis between neural trajectories
# see https://github.com/mitchellostrow/DSA/tree/main
dsa(epochs1, epochs2, n_delays=10, pca_components=10, verbose=False)
```
