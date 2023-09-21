import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import os
import sys
sys.path.append(str(root))

import wandb
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

#function to perform TSNE
def performTSNE(embeddings: np.ndarray,
                components: int, 
                lr: float = 150,
                perplexity: int = 30,
                angle: float = 0.2,
                verbose: int = 2) -> np.ndarray:
    
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding) is an unsupervised non-linear dimensionality
    reduction technique for data exploration and visualizing high-dimensional data. Non-linear 
    dimensionality reduction means that the algorithm allows us to separate data that cannot be
    separated by a straight line.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Args:
        X_reduced_dim: A reduced dimensional representation of an original image.
        n_components: The number of components to calculate.
        lr: The learning rate for t-SNE.
        perplexity: The perplexity is related to the number of expected nearest neighbors.
        angle: The tradeoff between speed and accuracy for Barnes-Hut T-SNE.
        verbose: Verbosity level.
    """

    # Create a T-SNE object or instance from the sklearn.manifold.TSNE class using the parameters
    tsne = TSNE(n_components=components,
                learning_rate=lr,
                perplexity=perplexity,
                angle=angle, 
                verbose=verbose)
    # Fit the T-SNE object to the data
    embeddings_tsne = tsne.fit_transform(embeddings)
    # Return the reduced embeddings_tsne to plot
    return embeddings_tsne

#Function to perform PCA
def performPCA(embeddings: np.ndarray,
                components: int) -> np.ndarray:
    # Create a PCA object or instance from the sklearn.decomposition.PCA class using the parameters
    pca = PCA(components, svd_solver='auto', whiten=True)
    pca.fit(embeddings)
    embeddings_pca = pca.transform(embeddings)
    
    return embeddings_pca

#Function to perform UMAP
def performUMAP(embeddings: np.ndarray,
                components: int,
                lr: float = 150,
                verbose: bool = True) -> np.ndarray:
    # Create the UMAP object or instance from the umap library using named parameters
    
    reducer = UMAP(n_components=components, learning_rate=lr, verbose=verbose, random_state=42)
    # Fit the UMAP object to the data
    embeddings_umap = reducer.fit_transform(embeddings)
    return embeddings_umap
    


'''
UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
     
     reducer = umap.UMAP(random_state=42)
     reducer.fit(digits.data)
     
     '''


def create_embeddings_component_table(num_components,
                                      embeddings_reduced: np.ndarray, 
                                      filenames: list,
                                      num_points: int)->pd.DataFrame:
    """
      Create a dataframe that contains the lower dimensional compenents of the embeddings 
      and the corresponding filenames.
    Args:
        embeddings_reduced: The reduced embeddings.
        filenames: The filenames of the images.
        num_points: The number of points to plot.
    Returns:
        A dataframe that contains the lower dimensional compenents of the embeddings 
        and the corresponding filenames.
    """
    # find the number of compents in the embeddings array by using the shape attribute
    print('type: ', type(embeddings_reduced))
    # create a panda series from the list of filenames
    filenames_series = pd.Series(filenames)
    # create a list of column names for the dataframe by using list comprehension and num_components
    df_columns = [f"CP_{i}" for i in range(num_components)]
    # append the file name to the dataframe df_columns (df_columns.append("filename"))
    df_columns.append("filename")
    # populate the dataframe with the embeddings_reduced and filenames_series
    df_components = pd.DataFrame(columns=df_columns, data=np.column_stack((embeddings_reduced[:num_points], filenames_series.iloc[:num_points])))
    # return the dataframe
    return df_components



# SIMSIAM CODE:
# transform embeddings to 2D using UMAP (dimension reduction algorithm)

# for the scatter plot we want to transform the images to a two-dimensional
# vector space using projections
# gaussian_2d = random_projection.GaussianRandomProjection(n_components=2, random_state=0)
# tsne_2d = TSNE(n_components=2, random_state=0)
# embeddings_2d = tsne_2d.fit_transform(embeddings)
# M = np.max(embeddings_2d, axis=0)
# m = np.min(embeddings_2d, axis=0)
# embeddings_2d = (embeddings_2d - m) / (M - m)

# gaussian = random_projection.GaussianRandomProjection(n_components=3, random_state=0)
# pca = PCA(n_components=3, random_state=0)
# tsne = TSNE(n_components=3, random_state=0)

# gaussian_embeddings = gaussian.fit_transform(embeddings)
# pca_embeddings = pca.fit_transform(embeddings)
# tsne_embedding = tsne.fit_transform(embeddings)


# # normalize the embeddings to fit in the [0, 1] square
# embed_list = [gaussian_embeddings, pca_embeddings, tsne_embedding]
# for i in range(len(embed_list)):
#     M = np.max(embed_list[i], axis=0)
#     m = np.min(embed_list[i], axis=0)
#     embed_list[i] = (embed_list[i] - m) / (M - m)