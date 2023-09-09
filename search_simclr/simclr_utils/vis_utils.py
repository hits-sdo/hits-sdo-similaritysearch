import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pyprojroot
import os
from datetime import datetime
from tqdm import tqdm

root = pyprojroot.here()
utils_dir = root/'search_utils'

import sys
sys.path.append(str(root))


def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, _, fnames in tqdm(dataloader):
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)


def plot_knn_examples(embeddings, filenames, path_to_data="root/data", n_neighbors=3, num_examples=6, vis_output_dir=None):
    """Plots multiple rows of random images with their nearest neighbors"""
    print(f'Embeddings.shape: {len(embeddings)}')
    # print("Inside plot_knn_examples() " + path_to_data)
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    print(f'len(samples_idx): {len(samples_idx)}')

    # loop through our randomly picked samples
    i = 0
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # print("Inside plot_knn_examples() loop")
            
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # get the correponding filename for the current index
            fname = os.path.join(path_to_data, filenames[neighbor_idx]) # tailor to our path directory structure
            print(f'file name: {fname}')
            # plot the image
            plt.imshow(get_image_as_np_array(fname)[:,:,0], cmap='hot')
            # set the title to the distance of the neighbor
            ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            # let's disable the axis
            plt.axis("off")
        #plt.show()
        i +=1
        plt.savefig(os.path.join(vis_output_dir, f'{i}_knn_plot.png'))
            # Save to file
        # if vis_output_dir is not None:
        #     now = datetime.now()
        #     now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            
        #     plt.savefig(os.path.join(vis_output_dir, f'{now_str}_knn_plot.png'))
                
           
# BELOW THIS LINE IS SIMSIAM CODE


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img
   
def plot_nearest_neighbors_3x3(example_image: str, i: int, embeddings, filenames, path_to_data):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    example_idx = filenames.index(example_image)
    # get distances to the cluster center
    distances = embeddings - embeddings[example_idx]
    distances = np.power(distances, 2).sum(-1).squeeze()
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = os.path.join(path_to_data, filenames[plot_idx])
        if plot_offset == 0:
            ax.set_title(f"Query Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
        else:
            plt.imshow(get_image_as_np_array(fname))
        # let's disable the axis
        plt.axis("off")





# ===========================


# def display_search_result(session_state, col2, embeddings_dict, data_path):
#     if 'All' in session_state["indices"]:
#         session_state["indices"] = [x for x in range(session_state['neighbors'])]

#     col2.write('Retrieved Images')
#     idx = embeddings_dict['filenames'].index(session_state['fnames'][0])
#     print('N:', embeddings_dict['embeddings'][idx, :5])

#     dim = math.ceil(math.sqrt(session_state['neighbors']))
#     fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
#     ax = ax.ravel()
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)

#     for a in ax:
#         a.axis('off')

#     for i, f in enumerate(session_state['fnames']):
#         img = cv2.imread(data_path + f)
#         ax[i].imshow(img[:, :, ::-1])

#         ax[i].text(10, 30, i, color='black', fontsize=(10/dim)*10)

#         if i in session_state['indices']:
#             overlay = cv2.rectangle(img, (0, 0), (127, 127), (0, 0, 255), 10)
#             ax[i].imshow(overlay[:, :, ::-1])

#     col2.pyplot(fig)

#     if st.button('Download Selected Images'):
#         with zipfile.ZipFile("selected_images.zip", "w") as zipf:
#             for n in session_state['indices']:
#                 # Add each file to the ZIP archive
#                 zipf.write(data_path+session_state['fnames'][n])

def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img