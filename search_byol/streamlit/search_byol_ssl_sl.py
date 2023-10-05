import streamlit as st
# from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from streamlit_cropper_ext import st_cropper_ext
import numpy as np
import re
import torch
# pull request to main for packager
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.byol_model import BYOL
from itertools import cycle

CSV_PATH = "/mnt/e/Downloads/MT-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs16_pds16_contrast_.zip"
CSV_32_PATH = "/mnt/d/Mis Documentos/AAResearch/SEARCH/best run/AMJ-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs32_pds32_contrast.zip"
# CSV_32_PATH = "/mnt/c/Users/jacob/Downloads/AMJ-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs32_pds32_contrast_.zip"
MODEL_32_PATH = "/mnt/d/Mis Documentos/AAResearch/SEARCH/best run/AMJ-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs32_pds32_contrast_.pt"
# MODEL_32_PATH = "/mnt/c/Users/jacob/Downloads/AMJ-ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs32_pds32_contrast_.pt"
TILES_PATH = "/mnt/d/Mis Documentos/AAResearch/SEARCH/hits-sdo-downloader/AIA211_193_171_256x256/AIA211_193_171_256x256"
# TILES_PATH = "/mnt/e/Downloads/AIA211_193_171_256x256"
PROJECTION_SIZE = 32
MAX_TILES_DISPLAYED = 50

def main():

    st.set_page_config(layout="wide")
    index_df = pd.read_csv(CSV_32_PATH, compression='zip')
    n_matches_max_value = index_df.shape[0]

    st.sidebar.title("HITS SDO Search Engine")
    st.sidebar.markdown("This search engine provides similar tiles according to the selected tile of the query image.")


    # Upload an image and set some options for demo purposes
    # st.header("Cropper Demo")
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    n_matches = st.sidebar.number_input(label="Number of Matches to Return", min_value=1, max_value=n_matches_max_value, value=15, step=1)
    box_color = st.sidebar.color_picker(label="Box Color", value='#FF00FF')
    #aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    # aspect_dict = {
    #     "1:1": (1, 1),
    #     "16:9": (16, 9),
    #     "4:3": (4, 3),
    #     "2:3": (2, 3),
    #     "Free": None
    # }
    aspect_ratio = (1, 1)
    model = load_byol_model(MODEL_32_PATH)
    # index_df = read_CSV_from_Zi, "x pixel coordinate bottom left"p(csv_path = "/mnt/e/Downloads/ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs16_pds16_contrast_.zip")
    
    # TODO: Placeholder for user-defined tile height and width; allow for custom tile sizes / selection of tile sizes.
    tile_height = 256
    tile_width = 256

    if img_file:
        # with col1:
        img = Image.open(img_file)
        # Get a cropped image from the frontend
        cropped_img = st_cropper_ext(img, realtime_update=True, box_color=box_color,
                                    aspect_ratio=aspect_ratio, max_height=700, max_width=700, tile_height=tile_height, tile_width=tile_width)

        # cols = cycle(st.columns())
        # Manipulate cropped image at will
        st.sidebar.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.sidebar.image(cropped_img, use_column_width=False)
        
        query_button = st.sidebar.button("Perform Query")
        # download_button = st.sidebar.button("Download CSV with Matches")
        filename = f'{n_matches}_matches.csv'        
        st.sidebar.download_button(
            label="Download CSV with Matches",
            data=open(filename),
            file_name=filename,
            mime='text/csv',
        )

    # with col2:

        transform = transforms.Compose([transforms.PILToTensor()])
        # only display max number of images
        if query_button:
            inference_image = transform(cropped_img)[None,:,:,:].float()/255     
            query_embedding = model.forward_momentum(inference_image)
            print(f"We did it! The inference is: {query_embedding}")
            kquery_neighbors = find_matches(query_embedding, index_df, PROJECTION_SIZE, n_matches=n_matches)
            
            print(kquery_neighbors)

            read_and_plot_matches(kquery_neighbors, PROJECTION_SIZE)

            create_csv_for_download(kquery_neighbors, tile_height=tile_height, tile_width=tile_width, filename=filename)

        # if download_button:

            

# Path to index for embeddings size 16 on Jake's computer:

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

    model = BYOL(projection_size=PROJECTION_SIZE, prediction_size=PROJECTION_SIZE)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# def read_CSV_from_Zip(csv_path = "/mnt/e/Downloads/ds1_bs64_lr0.1_doubleaug_ss0.1_se1.0_pjs16_pds16_contrast_.zip"):
#     index_df = pd.read_csv(csv_path, compression='zip')
#     return index_df

def find_matches(input_embedding : np.ndarray, index_df: pd.DataFrame, projection_size: int, n_matches: int = 15) -> pd.DataFrame: 
    """
        Function to find the closest n_matches for the provided embedding
        given an index_df

        parameters: 
            input_embedding: np.ndarray
                The embeddings of the user's cropped image after using model for inference

            index_df: pd.DataFrame
                Precomputed embeddings for all patches in the tile database with the relative path of each tile

            projection_size: int
                Size of the projection head in the model used to pre-calculate index_df
            
            n_matches: int
                The number of closest matches in the embedding space to return 

            
        returns:
            df_n_matches: pd.DataFrame
                A slice of the index_df containing the closest n_matches in the embedding space
    """

    # Calculate cosine sim on all embeddings compared to query_embedding
    index_df['cosine_similarity'] = cosine_similarity(index_df.iloc[:,:projection_size].values, input_embedding)

    # Get k nearest neighbors based on cosine_similarity
    query_neighbors = index_df.sort_values(by=['cosine_similarity'], ascending=False)
    kquery_neighbors = query_neighbors.iloc[:n_matches,:projection_size+1]
    
    return kquery_neighbors

def read_and_plot_matches(kquery_neighbors: pd.DataFrame, projection_size: int, n_columns: int = 10):
    number_of_rows = kquery_neighbors.shape[0]//n_columns + 1
    number_of_rows = np.min([MAX_TILES_DISPLAYED//n_columns, number_of_rows])
    for n_row in range(number_of_rows):
        cols = cycle(st.columns(n_columns)) # st.columns here since it is out of beta at the time I'm writing this
        i_start = n_row * n_columns
        i_end = (n_row + 1) * n_columns
        i_end = np.min([i_end, kquery_neighbors.shape[0]])
        filteredImages = kquery_neighbors.iloc[i_start:i_end, :]
        for idx in range(filteredImages.shape[0]):
            tile_match = Image.open(TILES_PATH + '/' + filteredImages.iloc[idx,projection_size])
            _ = tile_match.thumbnail((150,150))
            next(cols).image(tile_match)

    # for i in range(kquery_neighbors.shape[0]):
    #     st.image(tile_match)    


def create_csv_for_download(kquery_neighbors : pd.DataFrame, tile_height : int, tile_width : int, filename: str):
    # date_string = re.search(r"\d{8}_\d{6}(?!.+\d{8}_\d{6}.+)", data_filename).group().replace('_', 'T')
    df_download = pd.DataFrame()
    # df_embeddings = kquery_neighbors.copy()
    df_download['Date'] = kquery_neighbors['FileName'].apply(lambda x: re.search(r"\d{8}_\d{6}(?!.+\d{8}_\d{6}.+)", x).group().replace('_', 'T')) # retrieve date from filename.
    df_download['Date'] = df_download['Date'].apply(lambda x: '-'.join([x[0:4], x[4:6] ,x[6:]])) # format date.
    df_download['Date'] = df_download['Date'].apply(lambda x: ':'.join([x[0:13], x[13:15] , x[15:]]))
    df_download['JSOC_String'] = df_download['Date'].apply(lambda x: 'aia.lev1_euv_12s'+ f'[{x}]') # create JSOC string.
    df_download['X_Coordinate'] = kquery_neighbors['FileName'].apply(lambda x: re.split('_|\.', x)[-2]) # tile pixel coordinates.
    df_download['Y_Coordinate'] = kquery_neighbors['FileName'].apply(lambda x: re.split('_|\.', x)[-3])
    df_download['Tile_Height'] = tile_height
    df_download['Tile_Width'] = tile_width

    df_download = df_download.join(kquery_neighbors.iloc[:, 0:-1])
    # print(df_download['Date'], df_download['X_Coordinate'], df_download['Y_Coordinate'], df_download['JSOC_String'])
    df_download.to_csv(filename, index=False)
    

if __name__ == "__main__":
    main()