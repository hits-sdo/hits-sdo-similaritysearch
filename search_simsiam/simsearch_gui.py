""" -- TODO LIST --
# [X]Drag/drop an image for similarity search
# [ ] Add button for similarity search
# * on click call a function which does similarity search
# [ ] Sidebar with options
# * switching dataset
# * how many nearest neighbors
# * pick image from dataset
# * user upload their own dataset
# * euclidean vs cosine distance best image retrieval
# download similar images from the dataset and corresponding metadata
# [ ] set images to constant 1:1 and 128x128
# [ ] add option to select neighbor tiles and download with metadata (how does plotting images with pyplot affect ease of building selection)
# [ ] generate crop of query image for sim search
"""

import streamlit as st
from model import load_model
from nearest_neighbour_search import fetch_n_neighbor_filenames
import torchvision
from PIL import Image
import pickle
import math
import matplotlib.pyplot as plt
import cv2

if 'fnames' not in st.session_state:
    st.session_state['fnames'] = []

data_path = '/home/schatterjee/Documents/hits/AIA211_193_171_Miniset/'

@st.cache_resource
def simsiam_model(wavelength):
    # TODO add different model for each dataset and wavelength (and make pep8)
    return load_model('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=4-step=435.ckpt').eval()

@st.cache_resource
def embeddings_dict():
    return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict_new.p', 'rb'))


# image for similarity search
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["p", "jpg", "png"],
    key='img')

col1, col2 = st.columns([1, 2])

if st.session_state['img'] is not None:
    col1.write('Query Image')
    col1.image(st.session_state['img'], use_column_width=True)


def show_nearest_neighbours(wavelength,
                            num_images,
                            input_size,
                            dist_type):
    model = simsiam_model(wavelength)
    pil_image = Image.open(st.session_state['img'])

    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor()]
    )
    tensor = convert_tensor(pil_image)
    tensor = tensor[None, :3, :, :]

    embedding = model.backbone(tensor.to('cuda')).flatten(start_dim=1)
    query_embedding = embedding[0].cpu().data.numpy()
    print('Q:', query_embedding[:5])
    filenames = fetch_n_neighbor_filenames(query_embedding,
                                           embeddings_dict(),
                                           dist_type,
                                           num_images=num_images)

    st.session_state['fnames'] = filenames

    # for fname in filenames:
    #     col2.image(data_path + fname)


with st.sidebar:
    st.selectbox(
        'Wavelength',
        ('211 193 171', '211 193 171'),
        key='wavelength')

    st.selectbox(
        'Distance metric',
        ('Euclidean', 'Cosine'),
        key='dist')

    st.sidebar.slider('Number of Neighbours',
                      min_value=2,
                      max_value=50,
                      step=1,
                      value=26,
                      key='neighbors')

    st.sidebar.button('Perform Similarity Search',
                      on_click=show_nearest_neighbours,
                      args=([st.session_state['wavelength'],
                             st.session_state['neighbors'],
                             128,
                             st.session_state['dist']]),
                      key='search')


if st.session_state['search'] is True:
    col2.write('Retrieved Images')
    dim = math.ceil(math.sqrt(st.session_state['neighbors']))
    fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for a in ax:
        a.axis('off')

    for i, f in enumerate(st.session_state['fnames']):
        img = cv2.imread(data_path + f)
        ax[i].imshow(img[:, :, ::-1])

    col2.pyplot(fig)

    idx = embeddings_dict()['filenames'].index(st.session_state['fnames'][0])
    print('N:', embeddings_dict()['embeddings'][idx, :5])
