import sys,os
sys.path.append(os.getcwd())

import unittest
import h5py
from src.tiler import *

class TilerTest(unittest.TestCase):

    def setUp(self):
        self.test_file = 'data/test/HMI_magnetogram.20131025_000000_TAI.h5'
        self.test_data = h5py.File(self.test_file,'r')['magnetogram']
        self.test_data_height,self.test_data_width = np.shape(self.test_data)
        self.tile_dim = 100
        self.radius = 450
        self.parent_filename = '20131025_000000_hmi'
        self.output_dir = 'data/test_tiles'
        self.tiler = TilerClass(self.test_data,self.parent_filename,self.tile_dim,self.tile_dim,self.radius,(512,512),self.output_dir,{})

    def test_TilerSetup(self):
        self.assertGreaterEqual(self.tiler.parent_height,self.test_data_height)
        self.assertGreaterEqual(self.tiler.parent_width,self.test_data_width)

    def test_cutTiles(self):
        self.tiler.cut_set_tiles()
        self.assertEqual(len(self.tiler.tile_item_list),np.ceil(self.test_data_width/self.tile_dim)*np.ceil(self.test_data_height/self.tile_dim))
        self.assertIsInstance(self.tiler.tile_item_list[0],TileItem)
        tile = np.load(self.output_dir+'/'+self.parent_filename+'/tiles/'+self.tiler.tile_item_list[0].tile_fname)
        self.assertEqual(np.shape(tile)[0],self.tile_dim)
        self.assertEqual(np.shape(tile)[1],self.tile_dim)

    def test_cutSubsetTiles(self):
        self.tiler.cut_set_tiles(subset=True)
        self.assertLessEqual(len(self.tiler.tile_item_list),np.ceil(2*self.radius/self.tile_dim)*np.ceil(2*self.radius/self.tile_dim))
        tile = np.load(self.output_dir+'/'+self.parent_filename+'/tiles/'+self.tiler.tile_item_list[0].tile_fname)
        self.assertEqual(np.shape(tile)[0],self.tile_dim)
        self.assertEqual(np.shape(tile)[1],self.tile_dim)

    def test_padding(self):
        padding = (self.tiler.parent_height-self.test_data_height)//2
        self.assertEqual(np.sum(self.tiler.parent_image[:padding,:]),0)
        self.assertEqual(np.sum(self.tiler.parent_image[-padding:,:]),0)
        self.assertEqual(np.sum(self.tiler.parent_image[:,:padding]),0)
        self.assertEqual(np.sum(self.tiler.parent_image[:,-padding:]),0)

    def test_tileMetaDict(self):
        self.tiler.cut_set_tiles()
        self.tiler.generate_tile_metadata()
        self.assertIsInstance(self.tiler.tile_meta_dict,dict)
        self.assertEqual(self.tiler.tile_meta_dict['number_child_tiles'],len(self.tiler.tile_item_list))

    def test_exportJson(self):
        self.tiler.cut_set_tiles()
        self.tiler.generate_tile_metadata()
        self.tiler.convert_export_dict_to_json()
        self.assertTrue(os.path.exists(self.output_dir+os.sep+self.parent_filename+os.sep+'tile_meta_data/'+self.parent_filename+'_metadata.json'))

if __name__ == "__main__":
    unittest.main()