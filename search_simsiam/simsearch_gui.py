""" -- TODO LIST --
# [x] download similar images from the dataset with corresponding metadata
# [x] Add option to start of "select images index" to select All
# [x] Allow user to specify custom search space
# [x] add option for random augmentation to validate embeddings
# [x] add advanced menu to select between backbone and projection head
# [x] add option to clear augmentations
# [x] Speed up display of search results
# [x] switching model based on wavelength (use dictionary mapping name to model)
    # [ ] test with second 171 model
# [x] add smoother transitions
# [ ] crop functionality
    # [x] why did the aspect ratio stop working????
    # [ ] fix resetting of cropping on advanced option toggle
# [x] drag-and-drop not working...?

"""
import streamlit as st
import datetime
from copy import deepcopy


import numpy as np
from PIL import Image
from gui_utils import (
    root_path,
    empty_fnames,
    wavelengths_to_models,
    display_search_result,
    show_nearest_neighbors,
    embeddings_dict,
    apply_augmentation,
)
from streamlit_cropper import st_cropper


def box_algorithm(img: Image, aspect_ratio = False) -> dict:
    # Find a recommended box for the image (could be replaced with image detection)
    box = (cropped_coords[0],
           cropped_coords[1],
           cropped_coords[2],
           cropped_coords[3])
    box = [int(i) for i in box]
    height = box[3] - box[1]
    width = box[2] - box[0]

    # If an aspect_ratio is provided, then fix the aspect
    if aspect_ratio:
        ideal_aspect = aspect_ratio[0] / aspect_ratio[1]
        height = (box[3] - box[1])
        current_aspect = width / height
        if current_aspect > ideal_aspect:
            new_width = int(ideal_aspect * height)
            offset = (width - new_width) // 2
            resize = (offset, 0, -offset, 0)
        else:
            new_height = int(width / ideal_aspect)
            offset = (height - new_height) // 2
            resize = (0, offset, 0, -offset)
        box = [box[i] + resize[i] for i in range(4)]
        left = box[0]
        top = box[1]
        width = 0
        iters = 0
        while width < box[2] - left:
            width += aspect_ratio[0]
            iters += 1
        height = iters * aspect_ratio[1]
    else:
        left = box[0]
        top = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
    return {'left': int(left), 'top': int(top), 'width': int(width), 'height': int(height)}


st.set_page_config(
    page_title='SimSiam Similarity Search',
    page_icon='📚',
    layout='centered',
    initial_sidebar_state='auto'
    )

# define session state parameters
if 'fnames' not in st.session_state:
    empty_fnames(st.session_state)
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
if 'cropped_roi_tuple' not in st.session_state:
    st.session_state['cropped_roi_tuple'] = (int(1024*0.2), int(1024*0.2), int(1024*0.6), int(1024*0.6))
if 'indices' not in st.session_state:
    st.session_state['indices'] = [[]]

aug = 'perform_aug' in st.session_state and st.session_state['perform_aug']
clear = 'clear' in st.session_state and st.session_state['clear']

# upload image for similarity search
st.session_state['img'] = st.file_uploader(
    "Choose an image...",
    type=["p", "jpg", "png"],
    on_change=empty_fnames,
    args=([st.session_state]))

col1, col2 = st.columns([1, 2])  # define columns for images





# prevent every other crop from being ignored
# NOTE: adding the "if" and tabbing in cropped_coords means you have to crop twice to work,
# removing the "if" means crop does not persist through similarity search and augmentation

cropped_coords = (int(230 * 0.2), int(230 * 0.2), int(230 * 0.8), int(230 * 0.8))


with st.sidebar:
    wavelength_help = "Select dataset based on target wavelength(s).\
        🌊 Wavelengths are automatically mapped to RGB planes for color rendering."
    distance_metric_help = "Choose the distance metric used for nearest neighbor search.\
        Euclidean Distance is the L2 norm of the difference between two vectors.\
        The Cosine distance is the dot product between the two unit vectors "
    non_help = "Number of images retrieved in the search"
    pss_help = "Click to retrieve similar images"
    sii_help = "Select image index to download"
    embedding_help = "Select projection if you want augmentation invariance,\
        otherwise select backbone"

    st.selectbox(
        'Wavelength',
        wavelengths_to_models.keys(),
        key='wavelength',
        help=wavelength_help,
        on_change=empty_fnames,
        args=([st.session_state]))

    w = st.session_state['wavelength']
    data_path = root_path + f'AIA{w}_Miniset/'

    st.selectbox(
        'Filter with date',
        ('Yes', 'No'),
        index=1,
        key='search_type',
        on_change=empty_fnames,
        args=([st.session_state]))

    st.session_state['neighbors'] = st.slider(
        'Number of Neighbors',
        min_value=2,
        max_value=50,
        step=1,
        value=st.session_state['neighbors'],
        on_change=empty_fnames,
        args=([st.session_state]),
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
              key='sim_search',
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



    if len(st.session_state['fnames']) > 0:
        st.session_state['indices'].append(st.multiselect('Select image index:',
                                                      options=tuple(option),
                                                      help=sii_help))
        
        st.button('Clear all', key='clear', on_click=empty_fnames, args=([st.session_state]))
    else:
        if st.toggle("Advanced options", key='toggle', on_change=empty_fnames, args=([st.session_state])):
            st.session_state['dist'] = st.selectbox(
                'Distance metric',
                ('Euclidean', 'Cosine'),
                help=distance_metric_help,
                on_change=empty_fnames,
                args=([st.session_state]))

            st.session_state['embed'] = st.selectbox(
                'Embedding Source',
                ('Projection Head', 'Backbone'),
                help=embedding_help,
                on_change=empty_fnames,
                args=([st.session_state]))

            if st.button("Perform Random Augmentation", key='perform_aug', on_click=empty_fnames, args=([st.session_state])):
                st.session_state['augmented'] = 1
                apply_augmentation(st.session_state['img'])

            if st.button("Clear Augmentation", on_click=empty_fnames, args=([st.session_state])):
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
        img_container.image(st.session_state['aug_img'], use_column_width=True)
        pil_img = Image.fromarray((255*st.session_state['aug_img']).astype(np.uint8))

    if col1.toggle("Crop", on_change=empty_fnames, args=([st.session_state]), key='crop'):
        np_img = np.array(pil_img)
        aspect_ratio = np_img.shape[1]/np_img.shape[0]
        SIZE = 230
        pil_img = pil_img.resize((int(SIZE*aspect_ratio), SIZE))

        ind_comp = len(st.session_state['indices']) > 2 and st.session_state['indices'][-1] != st.session_state['indices'][-2]

        if st.session_state['sim_search'] or aug or ind_comp:

            cropped_coords = (st.session_state["cropped_roi_tuple"][0], st.session_state["cropped_roi_tuple"][1],
                              st.session_state["cropped_roi_tuple"][0]+st.session_state["cropped_roi_tuple"][2],
                              st.session_state["cropped_roi_tuple"][1]+st.session_state["cropped_roi_tuple"][3])
            scaling_factor = np_img.shape[0] / SIZE

            cropped_coords = tuple(
                [int(x / scaling_factor) for x in cropped_coords])

        c1 = deepcopy(cropped_coords)
        with col1:
            st.session_state['img'] = np_img[
                st.session_state['cropped_roi_tuple'][1]:st.session_state['cropped_roi_tuple'][1]
                    + st.session_state['cropped_roi_tuple'][3],
                st.session_state['cropped_roi_tuple'][0]:st.session_state['cropped_roi_tuple'][0] 
                    + st.session_state['cropped_roi_tuple'][2]
            ]

            cropped_roi = st_cropper(pil_img,
                                     realtime_update=True,
                                     box_color='#0000FF',
                                     return_type='box',
                                     box_algorithm=box_algorithm,
                                     should_resize_image=True
            )

            scaling_factor = np_img.shape[0] / SIZE
            cropped_roi_tuple = tuple(cropped_roi.values())

            cropped_coords = (cropped_roi_tuple[0], cropped_roi_tuple[1],
                              cropped_roi_tuple[0]+cropped_roi_tuple[2],
                              cropped_roi_tuple[1]+cropped_roi_tuple[3])

            c2 = deepcopy(cropped_coords)

            # scale all values
            cropped_roi_tuple = tuple(
                [int(x * scaling_factor) for x in cropped_roi_tuple]
                )

            st.session_state["cropped_roi_tuple"] = cropped_roi_tuple

            img = np_img[
                cropped_roi_tuple[1]:cropped_roi_tuple[1] + cropped_roi_tuple[3],
                cropped_roi_tuple[0]:cropped_roi_tuple[0] + cropped_roi_tuple[2]
            ]

            if c1 != c2:
                empty_fnames(st.session_state)
                st.session_state['img'] = img

            # Show Preview
            st.write("Cropped Query Image")
            st.image(img, use_column_width=True, clamp=True)
    else:
        cropped_coords = (230 * 0.2, 230 * 0.2, 230 * 0.8, 230 * 0.8)


if len(st.session_state['fnames']) > 0:
    display_search_result(st.session_state, col2, embeddings_dict(st.session_state, st.session_state['wavelength']), data_path)
