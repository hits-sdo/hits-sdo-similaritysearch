import math
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import zipfile


def display_search_result(session_state, col2, embeddings_dict, data_path):
    if 'All' in st.session_state["indices"]:
        st.session_state["indices"] = [x for x in range(st.session_state['neighbors'])]

    col2.write('Retrieved Images')
    idx = embeddings_dict()['filenames'].index(session_state['fnames'][0])
    print('N:', embeddings_dict()['embeddings'][idx, :5])

    dim = math.ceil(math.sqrt(session_state['neighbors']))
    fig, ax = plt.subplots(dim, dim, figsize=(10, 10))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for a in ax:
        a.axis('off')

    for i, f in enumerate(session_state['fnames']):
        img = cv2.imread(data_path + f)
        ax[i].imshow(img[:, :, ::-1])

        h, w, _ = img.shape

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
