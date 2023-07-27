import streamlit as st
# from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from streamlit_cropper import st_cropper
import numpy as np
import re
import torch
# pull request to main for packager
import torchvision.transforms as transforms

from models.byol_model import BYOL

def main():
    st.title("HITS SDO Selector")

    # Upload an image and set some options for demo purposes
    st.header("Cropper Demo")
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
    box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
    aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    aspect_dict = {
        "1:1": (1, 1),
        "16:9": (16, 9),
        "4:3": (4, 3),
        "2:3": (2, 3),
        "Free": None
    }
    aspect_ratio = aspect_dict[aspect_choice]
    model = load_byol_model()

    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                    aspect_ratio=aspect_ratio)
        
        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
        
    button = st.button("Run inference!")

    transform = transforms.Compose([transforms.PILToTensor()])

    if button:
        inference_image = transform(cropped_img)[None,:,:,:].float()      
        print(inference_image.shape)
        inf_image = model.forward_momentum(inference_image)
        print(f"We did it! The inference is: {inf_image}")

# Path to index for embeddings size 16 on Jake's computer:
# "/mnt/e/Downloads/ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs16_pds16_contrast_.zip"

    # file = st.file_uploader("Choose a file")
    # # image_array = np.asarray(file)
    # if file is not None:
        
        
        # image = Image.open(file)

        # st.image(image, caption='Our Image')

        # value = streamlit_image_coordinates("/mnt/e/bpsdata/AIA211_193_171_raw/20100601_000008_aia_211_193_171.jpg",
        #                                     height=1024, width=1024)
        # print(value)
        # st.write(value)

def load_byol_model(checkpoint_path='/mnt/c/Users/jacob/Downloads/MT-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs16_pds16_contrast_.pt'):
    """
        Function to load the BYOL pretrained model 
        parameters: 
            path: string
                path pointing at the location of the model 
            
        returns:
            model: pytorch model
                model initialized with pretrained weights 
    """

    model = BYOL(projection_size=16, prediction_size=16)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model






if __name__ == "__main__":
    main()