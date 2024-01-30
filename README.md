# meg_decoding_tools

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
scores, time_decod = decoding(meg_data=X, labels=y, decoder="logistic_linear", cv=4, ncores=8, verbose=False)
```

### Cross-temporal and cross-task generalisation

...

### Group-level hypothesis testing (based on default Bayes factors)

...

## State-space trajectories (latent module)

...

## Visualisation

...
