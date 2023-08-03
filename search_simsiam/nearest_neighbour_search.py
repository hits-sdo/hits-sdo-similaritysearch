import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_similarity


def fetch_n_neighbor_filenames(query_embedding, embeddings_dict, dist_type,
                               start_date=None, end_date=None, num_images=9):
    """Function to fetch filenames of nearest neighbors

    Args:
        query_embedding (np.ndarray): Embedding for query image.
        embeddings_dict (dict[filenames, embeddings]): Dictionary mapping filenames to embeddings.
        distance (str): Distance metric.
        num_images (int): Number of similar images to return. Defaults to 9.

    Returns:
        filenames: Filenames of images similar to the given embedding.

    """
    #distances = []
    # embeddings = np.array(list(embeddings_dict.values()))
    # filenames = list(embeddings_dict.keys())
    embeddings = embeddings_dict['embeddings']
    filenames = embeddings_dict['filenames']

    if dist_type.upper() == "EUCLIDEAN":
        distances = embeddings - query_embedding
        distances = np.power(distances, 2).sum(-1).squeeze()
    elif dist_type.upper() == "COSINE":
        distances = -1*cosine_similarity(embeddings,
                                         np.array([query_embedding]))
        distances = distances[:, 0]

    # Filter by date
    if start_date is not None and end_date is not None:
        # start_date = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')    #https://github.com/hits-sdo/hits-sdo-downloader/blob/main/search_download/downloader.py
        # end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')
        dates = np.array([datetime.datetime.strptime(filename.split('_')[0], '%Y%m%d').date() for filename in filenames])
        
        # my_datetime = datetime.datetime.combine(my_date, datetime.time(23, 59, 59))

        mask = (dates >= start_date) & (dates <= end_date) 
        distances = distances[mask]
        filenames = np.array(filenames)[mask]

    nn_indices = np.argsort(distances)[:num_images]
    nearest_neighbors = [filenames[idx] for idx in nn_indices]
    return nearest_neighbors
