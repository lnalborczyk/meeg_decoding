from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    cross_val_multiscore,
    LinearModel,
)


def time_decode(meg_data, labels, decoder="logistic", cv=4, ncores=-1, verbose=None):

    '''
    meg_data should be of shape items x channels x time_steps
    labels should be of length items
    '''

    # defining the decoder
    if decoder == "logistic":
        
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

    elif decoder == "logistic_linear":

        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver="liblinear")))
        
    elif decoder == "svc":

        clf = make_pipeline(StandardScaler(), SVC(C=1, kernel="linear"))

    # sliding the estimator on all time frames
    time_decod = SlidingEstimator(clf, n_jobs=ncores, scoring="roc_auc", verbose=verbose)

    # using N-fold cross-validation
    scores = cross_val_multiscore(time_decod, meg_data, labels, cv=cv, n_jobs=ncores, verbose=verbose)

    # returning these scores and the estimator
    return scores, time_decod


# function adapted from https://github.com/jdirani/MEGmvpa/blob/main/decoding.py
def cross_time_cond_gen(X_train, X_test, y_train, y_test, ncores=-1, verbose=None):

    '''
    training on X_train, testing on X_test
    with generalisation accross all time points
    meg_data should be of shape items x channels x time_steps
    labels should be of length items
    '''

    # defining the decoder
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    time_gen = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=ncores, verbose=verbose)

    # fitting it on the training data
    time_gen.fit(X=X_train, y=y_train)

    # scoring on the testing data
    scores = time_gen.score(X=X_test, y=y_test)

    # retrieving decision values
    decision_values = time_gen.decision_function(X_test)

    # retrieving predicted probs
    y_predicted_probs = time_gen.predict_proba(X_test)

    # returning everything
    return scores, decision_values, y_predicted_probs
