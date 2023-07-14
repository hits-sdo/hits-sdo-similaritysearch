import streamlit as st
from PIL import Image
import numpy as np
# pull request to main for packager 

def main():
    st.title("HITS SDO Selector")
    file = st.file_uploader("Choose a file")
    image_array = np.asarray(file)
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Our Image')

 

if __name__ == "__main__":
    main()