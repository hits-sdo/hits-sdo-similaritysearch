import glob
import pandas as pd
import numpy as np
import os

class Assembler():
    """
    Takes tile embeddings and reassembles into "image"s of size 
    tile_rows x tile_cols x embedding_dim
    """
    def __init__(self,tile_dim:int=128,img_dim:int=2048,embedding_dim:int=16,
                 data_path:str='data',run:str=''):
        self.tile_dim = tile_dim
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.data_path = data_path
        self.run = run
        self.files = sorted(glob.glob('wandb/*'+run+'/files/filenames*.csv'))

    def assemble_tiles(self,files,embeddings):
        """
        Take tile parent folder and assemble tiles into an "image" of size 
        tile_rows x tile_cols x embedding_dim

        Parameters:
            files (list):       list of tile filepaths
            embeddings (list):  list of corresponding embeddings 

        Returns:
            img (np array):     tile_rows x tile_cols x embedding_dim
        """
        img = np.zeros((self.img_dim//self.tile_dim,
                       self.img_dim//self.tile_dim,
                       self.embedding_dim))
        
        for file,embedding in zip(files,embeddings):
            file_split = file.split('/')[-1].strip('.npy').split('_')
            ind_row = int(file_split[1])//128
            ind_col = int(file_split[2])//128
            img[ind_row,ind_col,:] = embedding
                    
        return img


