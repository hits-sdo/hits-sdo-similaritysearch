""" -- TODO LIST --
# [ ] switching dataset
# [ ] download similar images from the dataset with corresponding metadata
# [ ] set images to constant 1:1 and 128x128
# [ x ] Add option to start of "select images index" to select All
# [ ] generate crop of query image for sim search ---
# [ ] find error cause of idx = embeddings_dict()['filenames'].index(st.session_state['fnames'][0])
# [ ] Allow user to specify custom search space
# [ ] Incorporate switching model based on dataset ---
# [ ] add viewing metadata of similar images ---
# [ ] add smoother transitions
# [ ] add option for random augmentation to validate embeddings ---
"""

# Potential usage of clickable_images custom streamlit plugin
# with col3:
#     data_urls = [f'data:image/jpeg;base64,{cv2.imread(data_path + f)}' for i, f in enumerate(st.session_state['fnames'])],

#     clicked = clickable_images(
#         paths=data_urls,
#         titles=[f"Image #{str(i)}" for i in range(len(st.session_state['fnames']))],
#         div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
#         img_style={"margin": "5px", "height": "200px"},
#     )

#     st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

import streamlit as st
import torchvision
import pickle
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import zipfile
import json
from model import load_model
from nearest_neighbour_search import fetch_n_neighbor_filenames
from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
from PIL import Image
from st_clickable_images import clickable_images
from io import BytesIO, BufferedReader
from streamlit_image_select import image_select


def empty_fnames():
    st.session_state['fnames'] = []


if 'fnames' not in st.session_state:
    empty_fnames()

data_path = '/home/schatterjee/Documents/hits/AIA211_193_171_Miniset/'


@st.cache_resource
def simsiam_model(wavelength):
    # TODO add different model for each dataset and wavelength (and make pep8)
    return load_model('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=4-step=435.ckpt').eval()


@st.cache_resource
def embeddings_dict():
    return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict_new.p', 'rb'))


# image for similarity search
st.session_state['img'] = st.file_uploader(
    "Choose an image...",
    type=["p", "jpg", "png"],
    on_change=empty_fnames)

col1, col2 = st.columns([1, 2])

if st.session_state['img'] is not None:
    img_text = col1.empty()
    img_text.write('Query Image')
    img_conainer = col1.empty()
    img_conainer.image(st.session_state['img'], use_column_width=True)

@st.cache_resource
def show_nearest_neighbours(wavelength,
                            num_images,
                            input_size,
                            dist_type,
                            start_date,
                            end_date):
    print("Showing the nearest neighbors")
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
                                           num_images=num_images,
                                           start_date=start_date,
                                           end_date=end_date)

    st.session_state['fnames'] = filenames


def apply_augmentation(img):
    '''
    Applys the current augmentation settings to the selected image
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
    st.session_state["img"] = img

    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # col1.pyplot(fig)

    img_text.write('Augmented Query Image')
    img_conainer.image(img, use_column_width=True)


def matin(arg1, arg2, arg3, arg4, arg5, arg6):
    """Easter egg funcion for testing"""
    print("Neighbors Slider Value: " + str(arg2))
    print("Dates: " + str(arg4))
    show_nearest_neighbours(arg1, arg2, arg3, arg4, arg5, arg6)


with st.sidebar:

    wavelength_help = "Select dataset based on target wavelegth(s). 🌊 Wavelengths are automatically mapped to RGB planes for color rendering."
    distancemetric_help = "Choose the distance metric used for nearest neighbor search. Euclidean Distance is the L2 norm of the difference between two vectors. The Cosine distance is the dot product between the two unit vectors "
    non_help = "Number of images retrieved in the search"
    pss_help = "Click to retrieve similar images"
    sii_help = "Select image index to download"

    st.selectbox(
        'Wavelength',
        ('211 193 171', '211 193 171'),
        key='wavelength',
        help=wavelength_help)

    st.selectbox(
        'Distance metric',
        ('Euclidean', 'Cosine'),
        key='dist',
        help=distancemetric_help)
    
    st.selectbox(
        'Filter with date',
        ('Yes', 'No'),
        index=1,
        key='search_type')

    if 'neighbors' not in st.session_state:
        st.session_state['neighbors'] = 26

    st.slider(
        'Number of Neighbours',
        min_value=2,
        max_value=50,
        step=1,
        value=st.session_state['neighbors'],
        key='neighbors',
        on_change=empty_fnames,
        help=non_help)


    # print("Current num neighbors: "+str(st.session_state['neighbors']))
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = None
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = None

    if st.session_state['search_type'] == 'Yes':
        st.subheader('Date Range')
        st.session_state['start_date'] = st.date_input("Begining of Time Range",
                      value=datetime.date(2011, 1, 1))
                    #   key='start_date')
        st.write('Begining of Time Range', st.session_state['start_date'])

        st.session_state['end_date'] = st.date_input("End of Time Range",
                      value=st.session_state['start_date'],
                      min_value=st.session_state['start_date'])
                    #   key='end_date')
        st.write('End of Time Range', st.session_state['end_date'])
        st.write(st.session_state['end_date'] > st.session_state['start_date'])

    else:
        st.session_state['start_date'] = None
        st.session_state['end_date'] = None

    st.button('Perform Similarity Search',
              on_click=show_nearest_neighbours,
              args=([st.session_state['wavelength'],
                     st.session_state['neighbors'],
                     128,
                     st.session_state['dist'],
                     st.session_state['start_date'],
                     st.session_state['end_date']]),
              help=pss_help)

    option = ['All'] + [x for x in range(st.session_state['neighbors'])]

    # if 'indices' not in st.session_state:
    #     st.session_state["indices"] = [x for x in range(st.session_state['neighbors'])] 

    st.session_state['indices'] = st.multiselect('Select image index:',
                                                 options=tuple(option),
                                                 help=sii_help)

    if st.button("Perform Random Augmentation"):
        apply_augmentation(st.session_state['img'])

if len(st.session_state['fnames']) > 0:
    if 'All' in st.session_state["indices"]:
        st.session_state["indices"] = [x for x in range(st.session_state['neighbors'])]

    col2.write('Retrieved Images')
    idx = embeddings_dict()['filenames'].index(st.session_state['fnames'][0])
    print('N:', embeddings_dict()['embeddings'][idx, :5])

    dim = math.ceil(math.sqrt(st.session_state['neighbors']))
    fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for a in ax:
        a.axis('off')

    for i, f in enumerate(st.session_state['fnames']):
        img = cv2.imread(data_path + f)
        ax[i].imshow(img[:, :, ::-1])

        h, w, _ = img.shape

        ax[i].text(10, 30, i, color='black', fontsize=(10/dim)*10)
        
        if i in st.session_state['indices']:
            overlay = cv2.rectangle(img, (0, 0), (127, 127), (0, 0, 255), 10)
            ax[i].imshow(overlay[:, :, ::-1])

    col2.pyplot(fig)

    if st.button('Download Selected Images'):
        with zipfile.ZipFile("selected_images.zip", "w") as zipf:
            for n in st.session_state['indices']:
                # Add each file to the ZIP archive
                zipf.write(data_path+st.session_state['fnames'][n])