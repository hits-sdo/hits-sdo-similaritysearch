import glob
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

class Assembler():
    """
    Takes tile embeddings and reassembles into "image"s of size 
    tile_rows x tile_cols x embedding_dim
    """
    def __init__(self,tile_dim:int=128,img_dim:int=2048,embedding_dim:int=16,
                 data_path:str='data',run:str='',datasets:list=['train','val']):
        self.tile_dim = tile_dim
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.run = run

        self.files = []
        self.embedding_files = []
        self.embeddings_proj_files = []
        for dataset in datasets:
            self.files.append(sorted(glob.glob('wandb/*'+run+'/files/filenames'+dataset+'.csv'))[-1])
            self.embedding_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings'+dataset+'.npy'))[-1])
            self.embeddings_proj_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings_proj'+dataset+'.npy'))[-1])
        
        self.df = pd.DataFrame()
        self.embeddings = []
        self.embeddings_proj = []

    def assemble_tiles(self,files,embeddings):
        """
        Take tile files and assembles into an "image" of size 
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
            ind_row = int(file_split[1])//self.tile_dim
            ind_col = int(file_split[2])//self.tile_dim
            img[ind_row,ind_col,:] = embedding
                    
        return img
    
    def reshape_embedding(self,embeddings,reducer,fit=True):
        """
        Reshape embedding to specified dimension either by some dimensionality reducer
        or by padding
        
        Parameters:
            embeddings (array):     embeddings to be reshaped
            reducer (sklearn like): has a fit_transform and a transform method
            fit (bool):             whether or not to fit the reducer

        Returns:
            embeddings (array):     reshaped to n samples x embedding dim 
        """
        if np.shape(embeddings)[1] > self.embedding_dim and fit:
            embeddings = reducer.fit_transform(embeddings)
        elif np.shape(embeddings)[1] > self.embedding_dim and not fit:
            embeddings = reducer.transform(embeddings)
        else:
            embeddings = np.pad(embeddings,((0,0),(0,self.embedding_dim-np.shape(embeddings)[1])))
        return embeddings

    def create_df(self):
        """
        Iterate through files and embeddings and compile into one dataframe/list
        """
        pca = PCA(n_components=self.embedding_dim)
        pca_proj = PCA(n_components=self.embedding_dim)

        for files,embeddings_file,embeddings_proj_file in zip(self.files,self.embedding_files,self.embeddings_proj_files):
            file_df = pd.read_csv(files)
            file_df['parent'] = file_df['filename'].str.split('/').str[-3]
            file_df['sample_time'] = file_df['filename'].str.split('/').str[-3].str.rsplit('_',n=1).str[0]

            embeddings = np.load(embeddings_file)
            embeddings_proj = np.load(embeddings_proj_file)

            if len(self.df)==0:
                self.df = file_df
                embeddings = self.reshape_embedding(embeddings,pca,fit=True)
                embeddings_proj = self.reshape_embedding(embeddings_proj,pca_proj,fit=True)
                self.embeddings = embeddings
                self.embeddings_proj = embeddings_proj

            else:
                self.df = pd.concat((self.df,file_df))
                embeddings = self.reshape_embedding(embeddings,pca,fit=False)
                embeddings_proj = self.reshape_embedding(embeddings_proj,pca_proj,fit=False)
                self.embeddings = np.concatenate((self.embeddings,embeddings),axis=0)
                self.embeddings_proj = np.concatenate((self.embeddings_proj,embeddings_proj),axis=0)

        self.df = self.df.reset_index(drop=True)

    def assemble_all(self):
        """
        To be run after create_df
        Assembles all tile embeddings, saves to data path and creates new index df
        """

        if len(self.df) == 0:
            print('Error, must run create_df first')
            return
        
        for parent in pd.unique(self.df['parent']):
            inds = self.df['parent']==parent
            files = self.df[inds]['filename']
            embeddings = self.embeddings[inds]
            embeddings_proj = self.embeddings_proj[inds]

            img = self.assemble_tiles(files,embeddings)
            img_proj = self.assemble_tiles(files,embeddings_proj)

            # save image
            np.save(self.data_path+os.sep+parent+'_embedding.npy',img)
            np.save(self.data_path+os.sep+parent+'_embedding_proj.npy',img_proj)

            self.df.loc[inds,'embedding_file'] = self.data_path+os.sep+parent+'_embedding.npy'
            self.df.loc[inds,'embedding_proj_file'] = self.data_path+os.sep+parent+'_embedding_proj.npy'

        print('Assembled',len(self.df),'tiles')
        self.df = self.df.drop_duplicates(subset='parent')
        print('Into',len(self.df),'embedded parent files')