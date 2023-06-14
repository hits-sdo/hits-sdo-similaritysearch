import sys,os
sys.path.append(os.getcwd())

import unittest
import h5py
from src.tile_items import *

class TilerTest(unittest.TestCase):

    def setUp(self):
        self.test_file = 'data/test/HMI_magnetogram.20131025_000000_TAI.h5'
        self.test_data = h5py.File(self.test_file,'r')['magnetogram']
        self.test_data_width,self.test_data_height = np.shape(self.test_data)
        self.tile_dim = 128
        self.tiler = TilerClass(self.test_data,self.test_file,self.tile_dim,self.tile_dim,485,(512,512),'data/test_tiles')

    def test_TilerSetup(self):
        self.assertEqual(self.tiler.parent_width,np.shape(self.test_data)[0])
        self.assertEqual(self.tiler.parent_height,np.shape(self.test_data)[1])
        self.assertEqual(len(self.tiler.tile_item_list),0)

    def test_cutTiles(self):
        self.tiler.cut_set_tiles()
        self.assertEqual(len(self.tiler.tile_item_list),np.ceil(self.test_data_width/self.tile_dim)*np.ceil(self.test_data_height/self.tile_dim))
        self.assertIsInstance(self.tiler.tile_item_list[0],TileItem)

if __name__ == "__main__":
    unittest.main()