import numpy as np
import torchvision
import datetime
import math
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
from model import load_model
from PIL import Image

import pickle

from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations


root_path = '/home/schatterjee/Documents/hits/'
model_path = root_path + '/hits-sdo-similaritysearch/search_simsiam/saved_model/'

wavelengths_to_models = {
    '171': model_path + 'epoch=8-step=14769.ckpt',  # TODO replace with correct model
    '211_193_171': model_path + 'epoch=9-step=17510.ckpt',
    '304_211_171': model_path + 'epoch=9-step=17510.ckpt',  # TODO replace with correct model
    '335_193_94': model_path + 'epoch=9-step=17510.ckpt'  # TODO replace with correct model
}

def apply_augmentation(img):
    '''
    Applies the current augmentation settings to the selected image
    checked if a user selected a region of interest -> cord_tup
    And displays the augmented image to the user
    '''
    img = Image.open(img)
    img = np.array(img)/255

    aug_list = AugmentationList(instrument="euv")
    aug_dict = aug_list.randomize()

    aug_img = Augmentations(img, aug_dict)
    fill_type = 'Nearest'
    img, _ = aug_img.perform_augmentations(fill_void=fill_type)
    st.session_state["aug_img"] = np.clip(img, 0, 1)
#     img_container.image(img, use_column_width=True)


def display_search_result(session_state, col2, embeddings_dict, data_path):
    if 'All' in session_state["indices"]:
        session_state["indices"] = [x for x in range(session_state['neighbors'])]

    col2.write('Retrieved Images')
    idx = embeddings_dict['filenames'].index(session_state['fnames'][0])
    print('N:', embeddings_dict['embeddings'][idx, :5])

    dim = math.ceil(math.sqrt(session_state['neighbors']))
    fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for a in ax:
        a.axis('off')

    for i, f in enumerate(session_state['fnames']):
        img = cv2.imread(data_path + f)
        ax[i].imshow(img[:, :, ::-1])

        ax[i].text(10, 30, i, color='black', fontsize=(10/dim)*10)

        if i in session_state['indices']:
            overlay = cv2.rectangle(img, (0, 0), (127, 127), (0, 0, 255), 10)
            ax[i].imshow(overlay[:, :, ::-1])

    col2.pyplot(fig)

    if st.button('Download Selected Images'):
        with zipfile.ZipFile("selected_images.zip", "w") as zipf:
            for n in session_state['indices']:
                # Add each file to the ZIP archive
                zipf.write(data_path+session_state['fnames'][n])


@st.cache_resource
def simsiam_model(wavelength):
    return load_model(wavelengths_to_models[wavelength]).eval()


def show_nearest_neighbors(session_state,
                           wavelength,
                           num_images,
                           input_size,
                           dist_type,
                           start_date,
                           end_date):
    print("Showing the nearest neighbors")
    model = simsiam_model(wavelength)
    if session_state['crop']:
        pil_image = Image.fromarray(session_state['img'].astype(np.uint8))
    elif session_state['augmented'] == 0:
        pil_image = Image.open(session_state['img'])
    else:
        pil_image = Image.fromarray((255*session_state['aug_img']).astype(np.uint8))

    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor()]
    )
    tensor = convert_tensor(pil_image)
    tensor = tensor[None, :3, :, :]

    embedding = model.backbone(tensor.to('cuda')).flatten(start_dim=1)

    if session_state['embed'] == 'Projection Head':
        embedding = model.projection_head(embedding)

    query_embedding = embedding[0].cpu().data.numpy()
    print('Q:', query_embedding[:5])
    filenames = fetch_n_neighbor_filenames(query_embedding,
                                           embeddings_dict(session_state, wavelength),
                                           dist_type,
                                           num_images=num_images,
                                           start_date=start_date,
                                           end_date=end_date)

    session_state['fnames'] = filenames


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


# def simsiam_model(wavelength):
#     # TODO add different model for each dataset and wavelength (and make pep8)
#     return load_model('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=9-step=17510.ckpt').eval()


def embeddings_dict(session_state, wavelength):
    if session_state['embed'] == 'Backbone':
        return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/'+f'embeddings_dict_{wavelength}.p', 'rb'))
    else:
        return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/'+f'embeddings_dict_{wavelength}_proj.p', 'rb'))
