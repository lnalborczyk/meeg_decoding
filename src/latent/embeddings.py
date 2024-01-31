
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


def plot_embeddings(points, points_color, labels, plot_title):
    
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white", constrained_layout=True)
    fig.suptitle(plot_title, size=16)
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=100, alpha=0.8)
    texts = []
    
    for x, y, s in zip(x, y, labels):
        texts.append(plt.text(x, y, s))
    
    adjust_text(texts, only_move={"points":"y", "texts":"y"}, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))


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
