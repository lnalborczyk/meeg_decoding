#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:01:48 2022
@author: christophe.gitton

And last extended/updated on January 19, 2024
@author: ladislas.nalborczyk@gmail.com
"""

import time
from tqdm import tqdm
import os
import os.path as op
import numpy as np
import statistics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib import cm
import glob
import pandas as pd
import multiprocessing as mp
import itertools
import mne
from mne import create_info
from mne.io import read_raw_fif, RawArray
from mne.preprocessing import ICA, find_bad_channels_maxwell
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator, apply_inverse, make_inverse_operator
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import csv
import seaborn as sns
import scipy
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import MDS
from adjustText import adjust_text
from scipy.io import loadmat
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
bf_package=importr("BayesFactor")

from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)

###############################################################################
# defining directories
###############################################################################

base_dir = "."
raw_dir = base_dir
fs_subjects_dir = op.join(base_dir, "IRM")
max_f_dir = op.join(base_dir, "derivatives/MAX_F_DATA")
ica_dir = op.join(base_dir, "derivatives/ICA")
sources_dir = op.join(base_dir, "derivatives/SRC")

###############################################################################
# compute average head position accross the recordings
###############################################################################

def my_raw_fname(subject, date, file):
    raw_fname = op.join(raw_dir, subject, date, file)
    return raw_fname


def my_mean_hp_fname(subject, date):
    mean_hp_fname = op.join(base_dir, max_f_dir, subject, date, "mean_hp.fif")
    return mean_hp_fname


def my_tsss_fname(subject, date, file):
    tsss_fname = op.join(max_f_dir, subject, date, op.splitext(file)[0] + "_trans_tsss.fif")
    return tsss_fname


def my_ica_fname(subject, date, file):
    tsss_fname = my_tsss_fname(subject, date, file)
    path, f = op.split(tsss_fname)
    ica_fname = op.join(ica_dir, subject, date, op.splitext(f)[0] + "_blink_cardio_corr.fif")
    return ica_fname


def mean_hp_compute(subject, date):
    mean_hp_fname = my_mean_hp_fname(subject, date)
    files = os.listdir(op.join(raw_dir, subject, date))
    files_to_process = [f for f in files if ".fif" in f and not any(
        ("empty" in f, "trans" in f, "sss" in f, "log" in f, "mean" in f, "bad" in f, "md5" in f))]
    i = 0
    for f in files_to_process:
        files_to_process[i] = op.join(raw_dir, subject, date, f)
        i = i+1
    path, file = op.split(mean_hp_fname)
    if not op.exists(path):
        os.makedirs(path)
    if not op.exists(mean_hp_fname):
        pos = np.zeros((len(files_to_process), 4, 4))
        i = 0
        for f in files_to_process:
            raw = read_raw_fif(f, allow_maxshield=True)
            pos[i] = raw.info["dev_head_t"]["trans"]
            i = i+1
        ch_names = raw.info["ch_names"]
        sfreq = raw.info["sfreq"]
        mean_hp_info = create_info(ch_names, sfreq=sfreq)
        mean_hp_info._unlocked=True
        mean_hp_info["dev_head_t"]["trans"] = np.mean(pos, axis=0)
        mean_hp_info["dig"] = raw.info["dig"]
        data = np.ones([len(ch_names), 1])
        mean_hp_raw = RawArray(data, mean_hp_info)
        mean_hp_raw.save(mean_hp_fname, overwrite=True)
    return files_to_process, mean_hp_fname


###############################################################################
# compute tsss with transposition to average head position
###############################################################################

def plotting_raw_data(subject, date, file):

    # importing the raw data file
    raw_fname = my_raw_fname(subject, date, file)
    raw = read_raw_fif(raw_fname, allow_maxshield=True, verbose=False).load_data()
    
    # launching the graphical interface (to look for bad channels)
    raw.plot()


###############################################################################
# compute tsss with transposition to average head position
###############################################################################

def tsss_compute(subject, date, file, bad_ch=None, auto_bad_ch=False):
    
    config_dir = op.join(raw_dir, subject, date, "sss_config")
    sss_cal = op.join(config_dir, "sss_cal.dat")
    ct_sparse = op.join(config_dir, "ct_sparse.fif")
    mean_hp_fname = my_mean_hp_fname(subject, date)
    if not op.exists(mean_hp_fname):
        mean_hp_compute(subject, date)
    raw_fname = my_raw_fname(subject, date, file)
    raw = read_raw_fif(raw_fname, allow_maxshield=True).load_data()

    if auto_bad_ch:
        
        # run find_bad_channels here
        # see https://mne.tools/stable/generated/mne.preprocessing.find_bad_channels_maxwell.html
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            #raw_check,
            raw,
            cross_talk=ct_sparse,
            calibration=sss_cal,
            return_scores=True,
            verbose=True,
        )
    
        # sanity checks
        print("automatically detected noisy channes:", auto_noisy_chs)
        print("automatically detected flat channels:", auto_flat_chs)
    
        # adding those to bad channels
        raw.info["bads"] = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
    
    if bad_ch:
        raw.info["bads"].extend(bad_ch)

    duration = len(raw.times)/1000
    raw_tsss = mne.preprocessing.maxwell_filter(
        raw, calibration=sss_cal, cross_talk=ct_sparse, st_duration=duration, destination=mean_hp_fname, verbose=True)
    tsss_fname = my_tsss_fname(subject, date, file)
    path, file = op.split(tsss_fname)
    
    if not op.exists(path):
        os.makedirs(path)
    raw_tsss.save(tsss_fname, overwrite=True)

    
###############################################################################
# compute tsss for all files
###############################################################################

def all_files_tsss_compute(subject, date, bad_ch, auto_bad_ch=False):
    files = os.listdir(op.join(raw_dir, subject, date))
    files_to_process = [f for f in files if ".fif" in f and not any(
        ("empty" in f, "trans" in f, "sss" in f, "log" in f, "mean" in f, "bad" in f, "md5" in f))]
    for file in files_to_process:
        tsss_compute(subject, date, file, bad_ch, auto_bad_ch)


###############################################################################
# compute ica and remove 1 blink comp and 1 ECG comp
###############################################################################

def compute_ICA(subject, date, file, ncores=8):
    # ICA decomposition
    tsss_fname = my_tsss_fname(subject, date, file)
    raw = mne.io.read_raw_fif(tsss_fname, preload=True)
    eog_ch_name = "BIO002"
    ecg_ch_name = "BIO003"
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1, h_freq=40)
    ica = ICA(n_components=0.95, max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    # find components to remove
    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(raw, eog_ch_name)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ecg_ch_name, method="correlation")
    n_max_eog, n_max_ecg = 1, 1
    eog_indices = eog_indices[:n_max_eog]
    ica.exclude = eog_indices
    ecg_indices = ecg_indices[:n_max_ecg]
    ica.exclude += ecg_indices
    # re-compute MEG signal with removing blink and cardio comps
    raw_copy = raw.copy()
    raw_ica_blink_cardio_corr = ica.apply(raw_copy)
    ica_fname = my_ica_fname(subject, date, file)
    if not op.exists(op.dirname(ica_fname)):
        os.makedirs(op.dirname(ica_fname))
    raw_ica_blink_cardio_corr.save(ica_fname, overwrite=True)
    
def all_files_ica_compute(subject, date):
    files = os.listdir(op.join(raw_dir, subject, date))
    print("found tsss files:", files)
    files_to_process = [f for f in files if ".fif" in f and not any(
        ("empty" in f, "training" in f, "log" in f, "mean" in f, "bad" in f, "md5" in f))]
    for file in files_to_process:
        compute_ICA(subject, date, file)
        
###############################################################################
# retrieving trigger codes
###############################################################################


def import_behavioural_data(subject, date):
    csv_file = glob.glob(op.join(subject, date) + "/*.csv", recursive=True)
    list_of_trials = pd.read_csv(filepath_or_buffer=csv_file[0], sep=";")
    
    return(list_of_trials)


###############################################################################
# getting epochs to be analysed (from all runs)
###############################################################################


def get_all_epochs(subject, date, blocks="auditory_block", lb=1, ub=120, decim=1):

    # importing behavioural data
    behav_data = import_behavioural_data(subject=subject, date=date)
    # removing training blocks
    behav_data = behav_data.iloc[110:,:]
    # retrieving the modality of each block
    blocks_modality = np.array(behav_data[["block", "modality"]].drop_duplicates()["modality"])
    # find indices of auditory or visual blocks
    blocks_indices = [i for i, x in enumerate(blocks_modality) if blocks in x]
    # sanity check
    # print("block_indices:", blocks_indices)
    # listing files
    files = os.listdir(op.join(raw_dir, subject, date))
    # keeping only block files
    files = [k for k in files if "run" in k]
    # ordering these files by name
    files.sort()
    # printing the name of the files
    print("found files:", files)
    # keeping only auditory or visual blocks
    files = [files[i] for i in blocks_indices]
    # printing the name of the files
    print("keeping only some blocks:", files)
    
    # initialising an empty list
    epochs_list = []
    
    # initialising the block counter
    block = 0

    for file in files: # for each file/block

        # printing progress
        print("Processing block number", block+1)
        print("Name of the file:", file)
        # retrieving the name of the ICA-processed file
        ica_fname = my_ica_fname(subject, date, file)
        # importing the raw (ICA-processed) data
        raw = mne.io.read_raw_fif(ica_fname, preload=True)
        # sanity checks
        # print("Channels' names:", raw.info["ch_names"])
        # filtering it between 1 and 120Hz
        # filt_raw = raw.filter(l_freq=1, h_freq=120, picks="meg")
        filt_raw = raw.filter(l_freq=lb, h_freq=ub, picks="meg")
        # retrieving events (i.e., triggers)
        # events = mne.find_events(raw=filt_raw, stim_channel="STI101", shortest_event=1)
        events = mne.find_events(raw=filt_raw, stim_channel="STI101", min_duration=0.002)
        # visualising events
        # mne.viz.plot_events(
        #     events, sfreq=filt_raw.info["sfreq"], first_samp=filt_raw.first_samp#, event_id=event_dict
        # )
        # keeping only the "stimulus_onset" events
        # events = mne.pick_events(events, include=9)
        # events = mne.pick_events(events=events, exclude=[51, 52, 53])
        # what are events 256 or 309?
        events = mne.pick_events(events=events, exclude=[51, 52, 53, 256, 309])
        # printing the number of events kept
        print(len(events), "events kept")
        
        # retrieving the behavioural data corresponding to the current block
        behav_data_current_block = behav_data.loc[behav_data["block"] == str(blocks_indices[block]+1)]

        # sanity checks
        # print("block_indices[block]+1:", str(blocks_indices[block]+1))
        # print(behav_data_current_block.head(10))
        
        # labelling these events using the behavioural data
        new_events, new_event_dict, new_behav_data_current_block = labelling_events(events=events, trials_list=behav_data_current_block)

        # sanity check
        # print(new_event_dict)
        
        # retrieving the behavioural data corresponding to the current block
        new_behav_data_current_block["category_id"] = new_behav_data_current_block["trigger_code"] + new_behav_data_current_block["target"] + new_behav_data_current_block["button_response"]

        # keeping only MEG channels (and excluding bad channels)
        picks = mne.pick_types(
            filt_raw.info, meg=True, eeg=False, stim=False, eog=False,
            #include=["MISC003", "MISC004"],
            exclude="bads"
        )

        # dictionary of reject criteria
        reject_criteria = dict(
            mag=6000e-15,  # 6000 fT
            grad=6000e-13  # 6000 fT/cm
            #mag=4000e-15,  # 4000 fT
            #grad=4000e-13  # 4000 fT/cm
            #eeg=150e-6,  # 150 µV
            #eog=250e-6, # 250 µV
        )
        
        # epoching these signals
        epochs_run = mne.Epochs(
            raw=filt_raw,
            events=new_events,
            # events=events,
            # event_id=new_event_dict,
            event_id=new_event_dict,
            tmin=-0.2,
            tmax=1.0,
            baseline=None,
            # baseline=(-0.2, 0.0),
            picks=picks,
            # preload=False,
            preload=True,
            # reject=dict(grad=4000e-13),
            reject=reject_criteria,
            flat=None, proj=False,
            decim=decim,
            reject_tmin=None, reject_tmax=None,
            detrend=None, on_missing="raise",
            reject_by_annotation=False,
            # this argument can be used to directly incorporate trial information
            # metadata=None,
            metadata=new_behav_data_current_block,
            event_repeated="error",
            verbose=None
        )

        # appending these events to the events' list
        epochs_list.append(epochs_run)

        # incrementing the block counter
        block += 1
        
    # concatenating the epochs from all runs
    epochs = mne.concatenate_epochs(
        epochs_list=epochs_list,
        add_offset=True,
        on_mismatch="raise",
        verbose=True
    )

    # returning these epochs
    return epochs
    

def labelling_events(events, trials_list, from_psychopy=False):

    if from_psychopy:
        
        # words with 3 letters and 2 syllables
        words_3l_2s = ["AGI", "OSÉ", "ÂGÉ", "ÉLU", "ÉMU", "BOA", "USÉ", "UNI", "GÉO", "ÉCU"]
    
        # words with 4 letters and 2 syllables
        words_4l_2s = ["ÉTAU", "DÉNI", "CURÉ", "COCA", "LAMA", "ÉTUI", "LOTO", "MENU", "COMA", "JURY"]
    
        # words with 7 letters and 2 syllables
        words_7l_2s = ["HAUTAIN", "PEUREUX", "MAUVAIS", "PENDANT", "COLLANT", "TONNEAU", "PINCEAU", "IMPLANT", "HONTEUX", "ROULEAU"]
    
        # acronyms with 3 letters and 3 syllables
        words_3l_3s = ["DVD", "EDF", "RER", "SDF", "SMS", "TGV", "USB", "BTS", "GPS", "QCM"]
    
        # words with 7 letters and 3 syllables
        words_7l_3s = ["MINIMAL", "DENSITÉ", "POSITIF", "ÉPICIER", "NATUREL", "BARBELÉ", "LÉOPARD", "ÉMOTION", "MÉDICAL", "PLACEBO"]
        
        # addding a column in the trial dataframe with an ID for each of these five categories
        conditions = [
            ([elem in words_3l_2s for elem in trials_list["item"]]),
            ([elem in words_4l_2s for elem in trials_list["item"]]),
            ([elem in words_7l_2s for elem in trials_list["item"]]),
            ([elem in words_3l_3s for elem in trials_list["item"]]),
            ([elem in words_7l_3s for elem in trials_list["item"]])
        ]
    
        # defining the resulting codes (one code per condition)
        results = [1, 2, 3, 4, 5]
        
        # assigning these codes
        trials_list["category_id"] = np.select(conditions, results, default=np.nan)
    
    # identifying targets (successive duplicated items)
    trials_list["target"] = 300*(trials_list["item"] == trials_list["item"].shift(1))
    
    # identifying button responses from the participant
    trials_list["button_response"] = 100*trials_list["response"]

    # sanity check
    # print(trials_list.head(20))
    
    # adding an identification code to event (1 code for each of the five categories of items)
    new_events = events
    # new_events[:,2] = new_events[:,2] + trials_list["category_id"]
    # new_events[:,2] = new_events[:,2] + trials_list["category_id"] + trials_list["target"] + trials_list["button_response"]
    new_events[:,2] = new_events[:,2] + trials_list["target"] + trials_list["button_response"]

    # new dictionary of trigger codes
    new_event_dict = {
        "3l3s/dvd": 1, "3l3s/edf": 2, "3l3s/rer": 3, "3l3s/sdf": 4, "3l3s/sms": 5,
        "3l3s/tgv": 6, "3l3s/usb": 7, "3l3s/bts": 8, "3l3s/gps": 9, "3l3s/qcm": 10,
        "3l2s/agi": 11, "3l2s/ose": 12, "3l2s/age": 13, "3l2s/elu": 14, "3l2s/emu": 15,
        "3l2s/geo": 16, "3l2s/use": 17, "3l2s/uni": 18, "3l2s/boa": 19, "3l2s/ecu": 20,
        "4l2s/etau": 21, "4l2s/deni": 22, "4l2s/cure": 23, "4l2s/coca": 24, "4l2s/lama": 25,
        "4l2s/etui": 26, "4l2s/loto": 27, "4l2s/menu": 28, "4l2s/coma": 29, "4l2s/jury": 30,
        "7l2s/hautain": 31, "7l2s/peureux": 32, "7l2s/mauvais": 33, "7l2s/pendant": 34, "7l2s/collant": 35,
        "7l2s/tonneau": 36, "7l2s/pinceau": 37, "7l2s/implant": 38, "7l2s/honteux": 39, "7l2s/rouleau": 40,
        "7l3s/minimal": 41, "7l3s/densite": 42, "7l3s/positif": 43, "7l3s/epicier": 44, "7l3s/naturel": 45,
        "7l3s/barbele": 46, "7l3s/leopard": 47, "7l3s/emotion": 48, "7l3s/medical": 49, "7l3s/placebo": 50
    }

    # sanity checks
    # print(trials_list.iloc[0:20,])
    # print(new_events[0:20,])

    # returning the new_events and new_event_dict objects
    return new_events, new_event_dict, trials_list
    
################################################################
# decoding functions
############################################################

# https://copyprogramming.com/howto/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
# checking and improving the trials-averaging function
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

    return new_epochs, new_labels


# defining a function to help with the pre-decoding processing
def prep_data_for_decoding(
    epochs, pca=True, n_components=50,
    moving_average=True, kernel_size=20,
    moving_average_with_decim=False, decim=4,
    trials_averaging=False, ntrials=4, shuffling_or_not=False
):

    # retrieving the MEG signals: n_epochs, n_meg_channels, n_times
    X = epochs.get_data()

    # retrieving the labels (i.e., items categories)
    # y = epochs.events[:, 2]
    y = abs((epochs.events[:, 2]-1) // 10)

    # some sanity checks
    # print("Labels:", y)
    print("Original shape of the MEG data:", X.shape)
    print("Length of the labels to be predicted:", len(y))

    if trials_averaging:

        # micro-averaging the MEG data (and the labels accordingly)
        X, y = trials_avg(epochs_data=X, labels=y, N=ntrials, shuffling_or_not=shuffling_or_not)
        print("Shape of the MEG data after trials averaging:", X.shape)
        # print("Shape of the labels after trials averaging:", y.shape)
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

        # select the number of dimensions so that explained variace is > 99%
        # pca = UnsupervisedSpatialFilter(PCA(n_components=0.99, svd_solver = 'full'), average=False)
        # pca = UnsupervisedSpatialFilter(PCA(n_components=n_components), average=False)
        # X = pca.fit_transform(X)
        # print("Shape of the MEG data after PCA:", X.shape)
        X = PCA_epochsArray(array=X, n_comp=n_components)


    return X, y


# defining a function to run the decoding (simple logistic regression) and returning the results
# using 8 parallel cores (to be adapted)
# see also https://github.com/jdirani/MEGmvpa
def decoding(meg_data, labels, decoder="logistic", cv=4, ncores=8, verbose=None):
    
    # making the pipeline
    # decoder can be "logistic" or "svc"
    if decoder=="logistic":
        
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

    elif decoder=="logistic_linear":

        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver="liblinear")))
        
    elif decoder=="svc":
        
        clf = make_pipeline(StandardScaler(), SVC(C=1, kernel="linear"))
    
    # sliding the estimator on all time frames
    time_decod = SlidingEstimator(clf, n_jobs=ncores, scoring="roc_auc", verbose=verbose)
    
    # here we use N-fold cross-validation
    scores = cross_val_multiscore(time_decod, meg_data, labels, cv=cv, n_jobs=ncores, verbose=verbose)
    
    # returning these scores
    return scores, time_decod


# function adapted from https://github.com/jdirani/MEGmvpa/blob/main/decoding.py
def cross_time_cond_gen(X_train, X_test, y_train, y_test, n_jobs=-1, verbose=False):
    
    '''Train on X_train, test on X_test with generalization accross all time points'''
    
    # creating the model
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=n_jobs, verbose=verbose)
    
    # fitting it
    time_gen.fit(X=X_train, y=y_train)
    
    # scoring on other condition
    scores = time_gen.score(X=X_test, y=y_test)

    # retrieving decision values
    decision_values = time_gen.decision_function(X_test)

    # retrieving predicted probs
    y_predicted_probs = time_gen.predict_proba(X_test)

    # returning everything
    return scores, decision_values, y_predicted_probs


# function adapted from https://github.com/jdirani/MEGmvpa/blob/main/decoding.py
def cross_time_cond_genCV(X_train, X_test, y_train, y_test, cv=4, n_jobs=-1):
    
    '''
    Train on X_train, test on X_test with generalization accross all time points
    Cross validation done by training on a subset of X_train and testing
    on a subset of X_test, with no repeats of trials accross the train-test split.
    Order of trials/exemplars in X_train, X_test, y_train, y_test must be the same.
    '''
    
    # creating the model
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=n_jobs, verbose=True)
    kf = KFold(n_splits=cv)

    scores = []
    
    for train_index, test_index in kf.split(X_train):
        
        fold_X_train = X_train[train_index]
        fold_X_test = X_test[test_index]

        fold_y_train = y_train[train_index]
        fold_y_test = y_test[test_index]

        # fitting
        time_gen.fit(X=fold_X_train, y=fold_y_train)
        
        # scoring on other condition
        fold_scores = time_gen.score(X=fold_X_test, y=fold_y_test)

        # appending the results
        scores.append(fold_scores)

    scores = np.array(scores)

    return scores


# defining a function to plot the decoding results
def plotting_decoding_scores(
    decoding_scores, x_ticks, plot_title, end_stim=0.2, plotting_theme="ticks",
    significant_clusters=None
):

    # reshaping these results in a pandas dataframe
    results_df = pd.DataFrame(decoding_scores.transpose())
    # results_df["time"] = epochs.times
    results_df["time"] = x_ticks
    results_df = pd.melt(results_df, id_vars="time", var_name="fold", value_name="score")
    results_df.head()
    
    # plotting theme
    sns.set_theme(style=plotting_theme)
    
    # plotting using sns
    fig, ax = plt.subplots(1, figsize=[12, 6])
    
    # plotting the chance level
    ax.axhline(y=0.5, color="k", ls="--", label="Chance level")
    
    # plotting the stimulus onset
    ax.axvline(x=0, color="k", ls="-")
    
    # plotting the stimulus duration (in visual blocks)
    ax.axvspan(0, end_stim, alpha=0.1, color="black")
    
    # plotting the average decoding accuracy (over folds) with 95% confidence interval
    sns.lineplot(
        data=results_df, x="time", y="score", ax=ax, lw=2,
        estimator="mean", errorbar=("ci", 95), n_boot=1000, label="Average accuracy"
    )
    
    # plotting timepoint significantly better than chance
    if significant_clusters is not None:
        
        for i in range(len(significant_clusters)):
            # ax.plot(epochs.times[significant_clusters_x[i]], significant_clusters_y, marker="o", color="b", markersize=5)
            ax.plot(epochs.times[significant_clusters[i]], np.min(np.mean(decoding_scores, axis=0)), marker="o", color="b", markersize=5)
    
    # filling accuracy above 0.5
    # plt.fill_between(x=results_df["time"], y1=0.5, y2=results_df["score"], alpha=0.3, where=results_df["score"]>0.5, interpolate=True)
    
    # specifying axis labels
    ax.set_title(plot_title, size=14, weight=800)
    ax.set_xlabel("Time (in seconds)", size=12)
    ax.set_ylabel("Decoding accuracy (AUC)", size=12)
    
    # adding a legend
    plt.legend(loc="upper right")

    # polishing the layout
    plt.tight_layout()


# defining a function to plot the generalisation across time
def plotting_gat(
    scores, x_ticks=None, epochs=None, plot_title="No title", xlab=None, ylab=None,
    chance=0.5, clim=None, colorbar=True, colorlab=None, diagonal=True
):
        
    # computing average decoding accuracy
    if len(scores.shape)>2:
        time_gen_avg_scores = np.mean(scores, axis=0)
    else:
        time_gen_avg_scores = scores
    
    # plotting the full (temporal generalisation) matrix
    fig, ax = plt.subplots(1, figsize=[12, 9])

    if clim is None:
        acc_min, acc_max = np.min(scores), np.max(scores)
        print("scores min and max:", acc_min, acc_max)
        max_range = max(np.abs(acc_min - chance), np.abs(acc_max - chance))
        clim = (chance - max_range, chance + max_range)

    if x_ticks is None:
        x_ticks=epochs.times[[0, -1, 0, -1]]
        
    im = ax.imshow(
        time_gen_avg_scores,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        #extent=epochs.times[[0, -1, 0, -1]],
        extent=x_ticks,
        vmin=np.min(clim), vmax=np.max(clim)
    )
    
    if xlab is not None:
        ax.set_xlabel(xlab)
    else:
        ax.set_xlabel("Testing time (in seconds)")
    
    if ylab is not None:
        ax.set_ylabel(ylab)
    else:
        ax.set_ylabel("Training time (in seconds)")

    ax.set_title(plot_title, fontweight="bold")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")

    if diagonal:

        ax.axline((1, 1), slope=1, color="red")
    
    if colorbar:
        
        cbar = plt.colorbar(im, ax=ax)

        if colorlab is not None:
            cbar.set_label(colorlab)
        else:
            cbar.set_label("Decoding accuracy (AUC)")

    # polishing the layout
    plt.tight_layout()


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


def stat_accuracy_single_ppt(scores, epochs, alpha_level=0.05, nsims=10, ncores=8):
    
    # shuffing labels to get null-hypothesis distribution of decoder accuracy over time
    # sould return "significant" time steps
    
    for i in range(nsims):
    
        # printing progress
        print("Simulation number", i+1, "out of", nsims)
        
        # preparing data (and labels) for decoding
        X, y = prep_data_for_decoding(
            epochs=epochs,
            pca=True, n_components=50,
            moving_average=True, kernel_size=5,
            trials_averaging=False, ntrials=4, shuffling_or_not=False
        )
    
        # shuffling the labels
        np.random.shuffle(y)
    
        # decoding time!
        decoding_scores = decoding(meg_data=X, labels=y, decoder="logistic", cv=4, ncores=ncores)
    
        # reshaping these results in a pandas dataframe
        temp_results_df = pd.DataFrame(decoding_scores.transpose())
        temp_results_df["time"] = epochs.times
        temp_results_df = pd.melt(temp_results_df, id_vars="time", var_name="fold", value_name="score")
        temp_results_df["simulation_id"] = i+1
    
        # appending results
        if i == 0:
    
            results = temp_results_df
    
        else:
    
            results = pd.concat([results, temp_results_df], ignore_index = True)


    # reshaping these results in a pandas dataframe
    observed_results_df = pd.DataFrame(scores.transpose())
    observed_results_df["time"] = epochs.times
    observed_results_df = pd.melt(observed_results_df, id_vars="time", var_name="fold", value_name="score")
    observed_results_df["simulation_id"] = 0

    # for each time step, compute the number of simulated accuracy curves equal or above to the observed one
    number_equal_or_above_across_time = []
    
    for j in range(len(epochs.times)):
    
        # retrieving data for the current time step
        current_time_step = results[(results.time == epochs.times[j])]
        current_time_step_observed = observed_results_df[(observed_results_df.time == epochs.times[j])]
    
        # averaging across CV folds
        averaged_score = current_time_step.groupby(["time", "simulation_id"]).mean()
        averaged_score_observed = current_time_step_observed.groupby(["time", "simulation_id"]).mean()
        averaged_score_observed2 = averaged_score_observed.loc[averaged_score_observed.index.repeat(len(averaged_score["score"]))]
        
        # computing the number of simulated decoding curves equal to or above the observed one
        logical_test = averaged_score["score"].reset_index(drop=True) >= averaged_score_observed2["score"].reset_index(drop=True)
        number_equal_or_above = np.sum(logical_test)
        
        # appending this number
        number_equal_or_above_across_time.append(number_equal_or_above)
        
    
    # retrieving "significant" time steps
    # that is, the time steps where observed accuracy was above 95% of simulated accuracies with shuffled labels
    significant_time_steps = np.where(np.array(number_equal_or_above_across_time) / nsims < alpha_level)[0]
    
    # returning the significan time steps
    return significant_time_steps
    

def plot_embeddings(points, points_color, labels, plot_title):
    
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white", constrained_layout=True)
    fig.suptitle(plot_title, size=16)
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=100, alpha=0.8)
    texts = []
    
    for x, y, s in zip(x, y, labels):
        texts.append(plt.text(x, y, s))
    
    adjust_text(texts, only_move={"points":"y", "texts":"y"}, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))


def perform_mds(x, ndim=2, normalise=True):

    mds = MDS(ndim, random_state=0, dissimilarity="precomputed")
    
    if normalise:
        
        embeddings = mds.fit_transform(minmax_scale(x))

    else:
        
        embeddings = mds.fit_transform(x)
        

    return embeddings


# https://www.binarystudy.com/2023/07/how-to-calculate-cosine-similarity-in-python.html
def computing_similarity(vector1, vector2):
    
    # computing the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    
    # computing the magnitudes (norms) of each vector
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # computing the cosine similarity using the dot product and vector norm
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    # returning it
    return cosine_similarity


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
    print('PCA explained var:%.3f'%explained_var)
    
    return pca_data


def PCA_epochsArray(array, n_comp):
    
    '''
    Fit PCA on epochs_fit, and transform epochs_trans using n_comp.
    Prints explained variance when using n_comp.
    '''
    
    mdl_PCA = PCA(n_comp)
    pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
    pca_data = pca.fit_transform(array)

    explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
    print('PCA explained variance:%.3f'%explained_var)
    
    return pca_data


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


def plot_neural_trajectories_2d(
    x_pca, x_pca_std=None, fs=1000, cmap="magma", add_line=False,
    plot_title="Neural trajectories through the entire trial", savefig=True
):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca.shape[0]) / (1 / fs)
    colormap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )
    
    # checking number of timepoints
    # print(len(timepoints))

    # plotting (normalised) variability as dot transparency
    if x_pca_std is None:
        
        scatter_alpha = 1

    else:

        x_pca_std_dims_average = np.mean(x_pca_std, axis=1)
        x_norm = (x_pca_std_dims_average - np.min(x_pca_std_dims_average)) / (np.max(x_pca_std_dims_average) - np.min(x_pca_std_dims_average))
        scatter_alpha = 1 - x_norm
        
    # plotting time
    fig = plt.figure(figsize = (12, 9) )
    scatter = plt.scatter(x_pca[:,0], x_pca[:,1], c = timepoints, cmap = colormap, norm = norm, alpha = scatter_alpha, s = 20 * scatter_alpha)
    if add_line: line = plt.plot(x_pca[:, 0], x_pca[:, 1], lw = 1, c = "black", alpha = 0.2)[0]

    # plotting crucial timepoints
    time_point = 0
    plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "red")
    print("200ms before stimulus onset (red):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    time_point = int(0.2 / (1 / fs))
    plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "green")
    print("Stimulus onset (green):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    time_point = int(0.4 / (1 / fs))
    plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "orange")
    print("Stimulus offset (orange):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    # defining axes labels etc
    plt.colorbar(scatter, label = "Time (s)")
    # cbar.ax.set_xticklabels(['Low', 'Medium', 'High']) 
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    # plt.title("PCA neural trajectories")
    plt.title(plot_title)
    plt.show()

    # saving the plot
    if savefig: plt.savefig(fname="tickertape02_c/neural_trajectories_2d.png", dpi=200, facecolor="white", transparent=False)


def plot_neural_trajectories_3d(x_pca, x_pca_std=None, fs=1000, cmap="magma", add_line=False, plot_title="Neural trajectories through the entire trial"):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca.shape[0]) / (1 / fs)
    colormap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )
    
    # checking number of timepoints
    # print(len(timepoints))

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
    #line = plt.plot(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], lw = 1, c = "black", alpha = 0.2)[0]
    
    ################################################################################
    # plotting markers for periods changes within the trial
    ########################################################################
    
    time_point = 0
    ax.scatter(x_pca[time_point, 0], x_pca[time_point, 1], x_pca[time_point, 2], marker = "o", s = 300, c = "red")
    print("200ms before stimulus onset (red):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    time_point = int(0.2 / (1 / fs) )
    ax.scatter(x_pca[time_point, 0], x_pca[time_point, 1], x_pca[time_point, 2], marker = "o", s = 300, c = "green")
    print("Stimulus onset (green):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    time_point = int(0.4 / (1 / fs) )
    ax.scatter(x_pca[time_point, 0], x_pca[time_point, 1], x_pca[time_point, 2], marker = "o", s = 300, c = "orange")
    print("Stimulus offset (orange):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    ax.view_init(20, 20, 0)
    plt.colorbar(sc, label = "Time (s)")
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(plot_title)
    
    plt.show()


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


def compare_neural_trajectories_2d(
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None, fs=1000,
    plot_title="Neural trajectories through the entire trial"
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

    # plotting crucial timepoints
    # time_point = 0
    # plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "red")
    # print("200ms before stimulus onset (red):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    # time_point = int(0.2 / 0.001)
    # plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "green")
    # print("Stimulus onset (green):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    # time_point = int(0.4 / 0.001)
    # plt.scatter(x_pca[time_point, 0], x_pca[time_point, 1], marker = "o", s = 300, c = "orange")
    # print("Stimulus offset (orange):", round(x_pca[time_point, 0], 3), round(x_pca[time_point, 1], 3), round(x_pca[time_point, 2], 3) )
    
    # defining axes labels etc
    plt.colorbar(scatter1, label = "Time in visual blocks (s)")
    plt.colorbar(scatter2, label = "Time in auditory blocks (s)")
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
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None, fs=1000,
    plot_title="Neural trajectories through the entire trial"
):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca1.shape[0]) / (1 / fs)
    # colormap = cm.get_cmap(cmap)
    colormap1 = cm.get_cmap("Blues")
    colormap2 = cm.get_cmap("Oranges")
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )
    
    # checking number of timepoints
    # print(len(timepoints))

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
    plt.colorbar(sc1, label = "Time in visual blocks (s)")
    plt.colorbar(sc2, label = "Time in auditory blocks (s)")
    
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
    x_pca1, x_pca2, x_pca_std1=None, x_pca_std2=None, fs=1000, time_interval=100,
    plot_title="Neural trajectories through the entire trial (visual trials in blue, auditory trials in orange)"
):
    
    # assigning colours to timepoints
    timepoints = np.arange(x_pca1.shape[0]) / (1 / fs)
    colormap1 = cm.get_cmap("Blues")
    colormap2 = cm.get_cmap("Oranges")
    norm = plt.Normalize(vmin = timepoints.min(), vmax = timepoints.max() )
    
    # checking number of timepoints
    # print(len(timepoints))

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

    # defining axes labels etc
    # plt.colorbar(sc1, label = "Time in visual blocks (s)")
    # plt.colorbar(sc2, label = "Time in auditory blocks (s)")
    
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.set_zlabel("PC3")
    
    # defining the plot's title
    # ax.set_title(plot_title)

    # polishing the layout
    # plt.tight_layout()

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

