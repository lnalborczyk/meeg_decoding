from DSA.dmd import DMD
from DSA.stats import *
from meeg.latent import compare_pca_through_time


def dsa(epochs1, epochs2, n_delays=10, pca_components=10, verbose=False):
    '''
    computing dynamical similarity between neural trajectories
    see https://github.com/mitchellostrow/DSA/tree/main
    '''
    
    # computing the PCA trajectories in a common space
    x_pca_contr1, x_pca_std_contr1, x_pca_contr2, x_pca_std_contr2 = compare_pca_through_time(epochs1, epochs2, n_components=pca_components)
    
    # fitting the DMD on the the average trajectory
    dmd_contr1 = DMD(x_pca_contr1, n_delays=n_delays, verbose=verbose)
    dmd_contr1.fit()
    
    # fitting the DMD on the average trajectory
    dmd_contr2 = DMD(x_pca_contr2, n_delays=n_delays, verbose=verbose)
    dmd_contr2.fit()
    
    # extracting the DMD matrices
    A_contr1 = dmd_contr1.A_v
    A_contr2 = dmd_contr2.A_v

    # comparing with SimilarityTransformDist
    comparison = SimilarityTransformDist(iters=1000, lr=0.001, verbose=verbose, device="cpu", group="O(n)")
    
    # fitting the two DMD matrices
    scores = comparison.fit_score(A_contr1, A_contr2)

    # printing the scores
    print("similarity:", scores)

    # returning this score
    return scores

