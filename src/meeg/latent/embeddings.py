import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.manifold import MDS
from adjustText import adjust_text


def perform_mds(x, ndim=2, normalise=True):

    mds = MDS(ndim, random_state=0, dissimilarity="precomputed")
    
    if normalise:
        
        embeddings = mds.fit_transform(minmax_scale(x))

    else:
        
        embeddings = mds.fit_transform(x)
        

    return embeddings


def compute_similarity(vector1, vector2):
    
    # computing the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    
    # computing the magnitudes (norms) of each vector
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # computing the cosine similarity using the dot product and vector norm
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    # returning it
    return cosine_similarity
