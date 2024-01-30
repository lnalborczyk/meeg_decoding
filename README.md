# meg_decoding_tools (in progress)

Open source tools for M/EEG preprocessing, basic analyses, and multivariate pattern analyses (aka decoding) based on [MNE-Python](https://mne.tools/stable/index.html).

# Installation

Clone this repository with `git clone https://github.com/lnalborczyk/meg_decoding_tools` and install using `python setup.py install` or `python setup.py develop` (creating symlinks to the source directory instead of installing locally), or install directly from Github with `pip install git+https://github.com/lnalborczyk/meg_decoding_tools`.

# Usage

## Preprocessing

...

## Decoding

### Decoding through time

```
import mne
import meg_decoding_tools as meg

from mne.decoding import (
    prep_data_for_decoding,
    decoding
)

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
scores, time_decod = decoding(meg_data=X, labels=y)
```

### Cross-temporal and cross-task generalisation

```
from mne.decoding import cross_time_cond_gen

# decoding time!
time_gen_scores, decision_values, y_predicted_probs = cross_time_cond_gen(X_train, X_test, y_train, y_test)
```

### Group-level hypothesis testing (based on default Bayes factors)

...

## Visualisation

### Decoding accuracy through time

```
from meg.plots import plotting_decoding_scores

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
from meg.plots import plotting_gat

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
