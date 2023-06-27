'''
    Unit tests for database class
'''
import sys,os
sys.path.append(os.getcwd())

import unittest
import glob
from datetime import datetime
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from src.assembler import Assembler
import torch

class AssemblerTest(unittest.TestCase):
    '''
        Test the data_loader class.
    '''
    def setUp(self):
        '''
            Setup the test environment.
        '''
        run = 'l0gz0vw6'
        self.embedding_dim = 16
        self.tile_files = np.array(pd.read_csv(sorted(glob.glob('wandb/*'+run+'/files/filenamestrain.csv'))[-1])['filename'])
        self.embeddings = np.load(sorted(glob.glob('wandb/*'+run+'/files/embeddingstrain.npy'))[-1])
        self.assembler = Assembler(embedding_dim=self.embedding_dim,
                                   run=run)

    def test_assemblerExists(self):
        self.assertIsInstance(self.assembler.tile_dim,int)
        self.assertIsInstance(self.assembler.run,str)
        self.assertIsInstance(self.assembler.data_path,str)
        self.assertIsInstance(self.assembler.files,list)

    def test_assembleTiles(self):
        pca = PCA(n_components=self.embedding_dim)
        embeddings_pca = pca.fit_transform(self.embeddings)
        img = self.assembler.assemble_tiles(self.tile_files,embeddings_pca)
        self.assertEqual(np.shape(img),(16,16,self.embedding_dim))
        self.assertGreater(np.sum(abs(img[:])),0)

if __name__ == "__main__":
    unittest.main()
