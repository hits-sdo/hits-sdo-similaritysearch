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
from src.data import TilesDataset,TilesDataModule
import torch

class DatasetTest(unittest.TestCase):
    '''
        Test the data_loader class.
    '''
    def setUp(self):
        '''
            Setup the test environment.
        '''
        self.data_path = 'data/tiles_HMI_small'
        self.data = TilesDataModule(self.data_path)

    def test_prepareData(self):
        self.data.prepare_data()
        self.assertIsInstance(self.data.df,pd.DataFrame)
        self.assertIsInstance(self.data.df['sample_time'][0],datetime)

    def tearDown(self):
        '''
            Discards the SDO database after test methods are called
        '''
        del self.data

if __name__ == "__main__":
    unittest.main()
