import numpy as np
import pandas as pd
import multiprocessing as mp
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
bf_package=importr("BayesFactor")


def compute_bf(data, bf_type=None):

    '''
    data should be a 2D numpy array of shape participants x time steps
    '''

    # extracting current cell and converting it to a dataframe
    df = pd.DataFrame(data)
    
    # converting it to R object
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(df)
        
    # computing the BF
    if bf_type == "notch":

        # computing the notched BF
        results=bf_package.ttestBF(x=r_data[0], mu=0, rscale="medium", nullInterval=[0.5, float("inf")])

        # storing the BF in favour of the accuracy being above chance versus chance level
        bf = np.asarray(r["as.vector"](results))[0]

    elif bf_type == "pos_vs_chance":
        
        results=bf_package.ttestBF(x=r_data[0], mu=0, rscale="medium", nullInterval=[0, float("inf")])
        # storing the BF in favour of the accuracy being above chance versus chance level
        bf = np.asarray(r["as.vector"](results))[0]
        
    elif bf_type == "pos_vs_neg":
        
        # storing the BF in favour of the accuracy being above chance versus below chance
        bf = np.asarray(r["as.vector"](results))[0] / np.asarray(r["as.vector"](results))[1]
    
    # returning the BF
    return bf


# this helper function is needed because map() can only be used for functions
def compute_bf_parallel(args):
    
    return compute_bf(*args)


# defining a function to compute BFs for differences with chance levels for a group of decoding accuracies over time
def bf_testing_time_decod(scores, chance=0.5):

    '''
    scores should be a 2D numpy array of shape participants x time steps
    '''
    
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
        
        results=bf_package.ttestBF(x=r_data[t], mu=0, rscale="medium", nullInterval=[0, float("inf")])
        bf.append(np.asarray(r["as.vector"](results))[0])


    # returning the BFs
    return bf


# defining a function to compute BFs for differences with chance level for a group of GAT matrices
def bf_testing_gat(scores, bf_type="pos_vs_chance", n_timepoints=None, chance=0.5, ncores=-1):
    
    # sanity check
    print("Shape of aggregated scores:", scores.shape)
        
    # retrieving the number of timepoints
    if n_timepoints is None:
        n_timepoints = scores.shape[2]
    
    # initialising an empty 2D array to store the BFs
    bf = np.zeros((n_timepoints, n_timepoints))

    # if sequential mode
    if ncores==1:

        # sanity check
        print("Sequential mode = -_-'")
        
        # looping over timepoints, converting decoding accuracy into effect size and computing the BF10
        for i in range(n_timepoints):
    
            # printing progress
            print("Processing row number", i+1, "out of", n_timepoints, "rows.")
            
            for j in range(n_timepoints):

                # computing and storing the BF
                bf[i, j] = compute_bf(data=scores[:, i, j]-chance, bf_type=bf_type)
        

    elif ncores > 1: # or if parallel mode

        # sanity check
        print("Parallel mode = \_ô_/")

        # initialising the progress bar
        # pbar = tqdm(total=n_timepoints**2)

        # creating a process pool that uses ncores cpus
        pool = mp.Pool(processes=ncores)
        
        # computing and storing the BF
        # https://stackoverflow.com/questions/29857498/how-to-apply-a-function-to-a-2d-numpy-array-with-multiprocessing
        bf = np.array(pool.map(compute_bf_parallel, ((scores[:, i, j]-chance, i, j) for i in range(n_timepoints) for j in range(n_timepoints)))).reshape(n_timepoints, n_timepoints)

        # sanity check
        print("Shape of bf object:", bf.shape)
        
        # closing the process pool
        pool.close()

        # waiting for all issued tasks to complete
        pool.join()
        
        # closing the progress bar
        # pbar.close()
    

    # returning the BFs
    return bf

