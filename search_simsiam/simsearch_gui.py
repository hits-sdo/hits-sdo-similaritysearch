""" -- TODO LIST --
# [x] download similar images from the dataset with corresponding metadata
# [x] Add option to start of "select images index" to select All
# [x] Allow user to specify custom search space
# [x] add option for random augmentation to validate embeddings
# [ ] switching model based on wavelength (use dictionary mapping name to model)
# [ ] add advanced menu to select between backbone and projection head
# [ ] add option to clear augmentations
# [ ] add smoother transitions
"""
import streamlit as st
import torchvision
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from model import load_model
from nearest_neighbour_search import fetch_n_neighbor_filenames
from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
from PIL import Image

from gui_utils import display_search_result

def empty_fnames():
    st.session_state['fnames'] = []


if 'fnames' not in st.session_state:
    empty_fnames()

data_path = '/home/schatterjee/Documents/hits/AIA211_193_171_Miniset/'


@st.cache_resource
def simsiam_model(wavelength):
    # TODO add different model for each dataset and wavelength (and make pep8)
    return load_model('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/saved_model/epoch=9-step=17510.ckpt').eval()


# @st.cache_resource
def embeddings_dict():
    if st.session_state['embed'] == 'Backbone':
        return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict_211_193_171.p', 'rb'))
    else:
        return pickle.load(open('/home/schatterjee/Documents/hits/hits-sdo-similaritysearch/search_simsiam/embeddings_dict_211_193_171_proj.p', 'rb'))

# image for similarity search
st.session_state['img'] = st.file_uploader(
    "Choose an image...",
    type=["p", "jpg", "png"],
    on_change=empty_fnames)

col1, col2 = st.columns([1, 2])

if st.session_state['img'] is not None:
    img_text = col1.empty()
    
    img_conainer = col1.empty()
    if st.session_state['augmented'] == 0:
        img_text.write('Query Image')
        img_conainer.image(st.session_state['img'], use_column_width=True)
    else:
        img_text.write('Augmented Image')
        img_conainer.image(st.session_state['aug_img'], use_column_width=True)


if 'dist' not in st.session_state:
    st.session_state['dist'] = 'Euclidean'
if 'embed' not in st.session_state:
    st.session_state['embed'] = 'Projection Head'
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = None
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = None
if 'augmented' not in st.session_state:
    st.session_state['augmented'] = 0
if 'aug_img' not in st.session_state:
    st.session_state['aug_img'] = None

# @st.cache_resource
def show_nearest_neighbours(wavelength,
                            num_images,
                            input_size,
                            dist_type,
                            start_date,
                            end_date):
    print("Showing the nearest neighbors")
    model = simsiam_model(wavelength)
    if st.session_state['augmented'] == 0:
        pil_image = Image.open(st.session_state['img'])
    else:
        pil_image = Image.fromarray((255*st.session_state['aug_img']).astype(np.uint8))

    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor()]
    )
    tensor = convert_tensor(pil_image)
    tensor = tensor[None, :3, :, :]

    embedding = model.backbone(tensor.to('cuda')).flatten(start_dim=1)

    if st.session_state['embed'] == 'Projection Head':
        embedding = model.projection_head(embedding)
        

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
    st.session_state["aug_img"] = img

    img_text.write('Augmented Query Image')
    img_conainer.image(img, use_column_width=True)


with st.sidebar:

    wavelength_help = "Select dataset based on target wavelegth(s). 🌊 Wavelengths are automatically mapped to RGB planes for color rendering."
    distancemetric_help = "Choose the distance metric used for nearest neighbor search. Euclidean Distance is the L2 norm of the difference between two vectors. The Cosine distance is the dot product between the two unit vectors "
    non_help = "Number of images retrieved in the search"
    pss_help = "Click to retrieve similar images"
    sii_help = "Select image index to download"
    embedding_help = "Select projection if you want augmentation invariance, otherwise select backbone"

    st.selectbox(
        'Wavelength',
        ('211 193 171', '211 193 171'),
        key='wavelength',
        help=wavelength_help)

    st.selectbox(
        'Filter with date',
        ('Yes', 'No'),
        index=1,
        key='search_type')

    if 'neighbors' not in st.session_state:
        st.session_state['neighbors'] = 26

    st.session_state['neighbors'] = st.slider(
        'Number of Neighbours',
        min_value=2,
        max_value=50,
        step=1,
        value=st.session_state['neighbors'],
        on_change=empty_fnames,
        help=non_help)

    if st.session_state['search_type'] == 'Yes':
        st.subheader('Date Range')
        st.session_state['start_date'] = st.date_input(
            "Begining of Time Range",
            value=datetime.date(2011, 1, 1))

        st.write('Begining of Time Range', st.session_state['start_date'])

        st.session_state['end_date'] = st.date_input(
            "End of Time Range",
            value=st.session_state['start_date'],
            min_value=st.session_state['start_date'])

        st.write('End of Time Range', st.session_state['end_date'])
        st.write(st.session_state['end_date'] > st.session_state['start_date'])

    else:
        st.session_state['start_date'] = None
        st.session_state['end_date'] = None

    st.sidebar.write(st.session_state['embed'])
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

    st.session_state['indices'] = st.multiselect('Select image index:',
                                                 options=tuple(option),
                                                 help=sii_help)

    if st.button("Perform Random Augmentation"):
        st.session_state['augmented'] = 1
        apply_augmentation(st.session_state['img'])

    if st.toggle("Advanced options"):
        st.session_state['dist'] = st.selectbox(
            'Distance metric',
            ('Euclidean', 'Cosine'),
            help=distancemetric_help)

        st.session_state['embed'] = st.selectbox(
            'Embedding Source',
            ('Projection Head', 'Backbone'),
            help=embedding_help)
            



if len(st.session_state['fnames']) > 0:
    display_search_result(st.session_state, col2, embeddings_dict, data_path)
