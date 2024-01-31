# meg_decoding_tools (work in progress)

Open source tools for M/EEG preprocessing, basic analyses, and multivariate pattern analyses (aka decoding) based on [MNE-Python](https://mne.tools/stable/index.html).

# Installation

<!--

Clone this repository with `git clone https://github.com/lnalborczyk/meg_decoding_tools` and install using `python setup.py install` or `python setup.py develop` (creating symlinks to the source directory instead of installing locally).

Or install directly from Github with `pip install git+https://github.com/lnalborczyk/meg_decoding_tools.git`.

-->

Clone this repository with `git clone https://github.com/lnalborczyk/meg_decoding_tools` and install using `python3 -m pip install .`

Or install directly from Github with `pip install git+https://github.com/lnalborczyk/meg_decoding_tools`.

# Usage

Function from this module assume that you have some M/EEG data that is properly filtered, resampled, and epoched using MNE (i.e., it does not cover preprocessing).

## Decoding

### Decoding through time

```
# importing mne and sub-packages from meg_decoding_tools
import mne
from decoding.decode import time_decode
from decoding.prepare import prep_data_for_decoding

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
from decoding.decode import cross_time_cond_gen

# decoding time!
time_gen_scores, decision_values, y_predicted_probs = cross_time_cond_gen(X_train, X_test, y_train, y_test)
```

### Group-level hypothesis testing (based on default Bayes factors)

...

## Visualisation

### Decoding accuracy through time

```
from plots.scores import plotting_decoding_scores

# plotting the decoding accuracy over time
plotting_decoding_scores(
    decoding_scores=scores,
    x_ticks=decoding_epochs.times,
    end_stim=0.2,
    plot_title="Sensor space decoding"
)
```

### Decoding accuracy through time with BFs

...

### Decoding generalisation

```
from plots.scores import plotting_gat

plotting_gat(
    scores=time_gen_scores,
    epochs=decoding_epochs,
    plot_title="Temporal generalisation matrix"
)
```

### Decoding generalisation with BFs

...

## State-space trajectories (latent module)

...
