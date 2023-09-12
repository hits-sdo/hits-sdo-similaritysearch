""" -- TODO LIST --
# [x] download similar images from the dataset with corresponding metadata
# [x] Add option to start of "select images index" to select All
# [x] Allow user to specify custom search space
# [x] add option for random augmentation to validate embeddings
# [x] add advanced menu to select between backbone and projection head
# [x] add option to clear augmentations
# [ ] Speed up display of search results
# [x] switching model based on wavelength (use dictionary mapping name to model)
    # [ ] test with second 171 model
# [ ] add smoother transitions
# [ ] crop functionality
    # [ ] reorder so crop -> augment instead of augment -> crop
# [ ] drag-and-drop not working...?

"""
import streamlit as st
import datetime
import numpy as np
from PIL import Image
from model import load_model
from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
from gui_utils import (
    root_path, 
    model_path,
    wavelengths_to_models,
    display_search_result,
    show_nearest_neighbors,
    embeddings_dict
)
from streamlit_cropper import st_cropper





def empty_fnames():
    st.session_state['fnames'] = []


# define session state parameters
if 'fnames' not in st.session_state:
    empty_fnames()
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
if 'neighbors' not in st.session_state:
    st.session_state['neighbors'] = 26

# upload image for similarity search
st.session_state['img'] = st.file_uploader(
    "Choose an image...",
    type=["p", "jpg", "png"],
    on_change=empty_fnames)

col1, col2 = st.columns([1, 2])  # define columns for images


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
    st.session_state["aug_img"] = img

    img_container.image(img, use_column_width=True, clamp=True)


with st.sidebar:
    wavelength_help = "Select dataset based on target wavelength(s). 🌊 Wavelengths are automatically mapped to RGB planes for color rendering."
    distance_metric_help = "Choose the distance metric used for nearest neighbor search. Euclidean Distance is the L2 norm of the difference between two vectors. The Cosine distance is the dot product between the two unit vectors "
    non_help = "Number of images retrieved in the search"
    pss_help = "Click to retrieve similar images"
    sii_help = "Select image index to download"
    embedding_help = "Select projection if you want augmentation invariance, otherwise select backbone"

    st.selectbox(
        'Wavelength',
        wavelengths_to_models.keys(),
        key='wavelength',
        help=wavelength_help,
        on_change=empty_fnames)
    
    w = st.session_state['wavelength']
    data_path = root_path + f'AIA{w}_Miniset/'

    st.selectbox(
        'Filter with date',
        ('Yes', 'No'),
        index=1,
        key='search_type',
        on_change=empty_fnames)

    st.session_state['neighbors'] = st.slider(
        'Number of Neighbors',
        min_value=2,
        max_value=50,
        step=1,
        value=st.session_state['neighbors'],
        on_change=empty_fnames,
        help=non_help)

    if st.session_state['search_type'] == 'Yes':
        st.subheader('Date Range')
        st.session_state['start_date'] = st.date_input(
            "Beginning of Time Range",
            value=datetime.date(2011, 1, 1))

        st.write('Beginning of Time Range', st.session_state['start_date'])

        st.session_state['end_date'] = st.date_input(
            "End of Time Range",
            value=st.session_state['start_date'],
            min_value=st.session_state['start_date'])

        st.write('End of Time Range', st.session_state['end_date'])
        st.write(st.session_state['end_date'] > st.session_state['start_date'])

    else:
        st.session_state['start_date'] = None
        st.session_state['end_date'] = None

    st.button('Perform Similarity Search',
              on_click=show_nearest_neighbors,
              args=([st.session_state,
                     st.session_state['wavelength'],
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

    if st.toggle("Advanced options"):
        st.session_state['dist'] = st.selectbox(
            'Distance metric',
            ('Euclidean', 'Cosine'),
            help=distance_metric_help,
            on_change=empty_fnames)

        st.session_state['embed'] = st.selectbox(
            'Embedding Source',
            ('Projection Head', 'Backbone'),
            help=embedding_help,
            on_change=empty_fnames)

        if st.button("Perform Random Augmentation", on_click=empty_fnames):
            st.session_state['augmented'] = 1

        if st.button("Clear Augmentation", on_click=empty_fnames):
            st.session_state['augmented'] = 0

if st.session_state['img'] is not None:
    img_text = col1.empty()

    img_container = col1.empty()
    if st.session_state['augmented'] == 0:
        img_text.write('Query Image')
        img_container.image(st.session_state['img'], use_column_width=True)
        pil_img = Image.open(st.session_state['img'])
    else:
        img_text.write('Augmented Image')
        apply_augmentation(st.session_state['img'])
        img_container.image(st.session_state['aug_img'], use_column_width=True)
        pil_img = Image.fromarray((255*st.session_state['aug_img']).astype(np.uint8))

    if col1.toggle("Crop", on_change=empty_fnames, key='crop'):
        with img_container:
            np_img = np.array(pil_img)
            aspect_ratio = np_img.shape[1]/np_img.shape[0]
            SIZE = 230
            pil_img = pil_img.resize((int(SIZE*aspect_ratio), SIZE))

            cropped_coords = st_cropper(pil_img,
                                        realtime_update=True,
                                        box_color='#0000FF',
                                        return_type='box',
                                        should_resize_image=True,
                                        aspect_ratio=None)

            scaling_factor = np_img.shape[0] / SIZE
            cord_tuple = tuple(map(int, cropped_coords.values()))

            # scale all values
            cord_tuple = tuple(
                [int(x * scaling_factor) for x in cord_tuple]
                )

            st.session_state['img'] = np_img[
                cord_tuple[1]:cord_tuple[1] + cord_tuple[3],
                cord_tuple[0]:cord_tuple[0] + cord_tuple[2]
            ]

            # Show Preview
            col1.write("Cropped Image")
            col1.image(st.session_state['img'], use_column_width=True, clamp=True)

if len(st.session_state['fnames']) > 0:
    # embeddings_dict = embeddings_dict(st.session_state)
    display_search_result(st.session_state, col2, embeddings_dict(st.session_state, st.session_state['wavelength']), data_path)
