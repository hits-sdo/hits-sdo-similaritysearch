import streamlit as st
# from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from streamlit_cropper import st_cropper
import numpy as np
# pull request to main for packager 

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
    
    

    # file = st.file_uploader("Choose a file")
    # # image_array = np.asarray(file)
    # if file is not None:
        
        
        # image = Image.open(file)

        # st.image(image, caption='Our Image')

        # value = streamlit_image_coordinates("/mnt/e/bpsdata/AIA211_193_171_raw/20100601_000008_aia_211_193_171.jpg",
        #                                     height=1024, width=1024)
        # print(value)
        # st.write(value)
 

if __name__ == "__main__":
    main()