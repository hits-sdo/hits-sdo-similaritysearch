import sys,os
sys.path.append(os.getcwd())

import h5py
import numpy as np
import os
import json
# import pyprojroot
from dataclasses import dataclass
from search_utils.image_utils import *
from typing import NamedTuple 
from PIL import Image

class TileItem(NamedTuple):
    tile_width:int
    tile_height:int
    origin_row: int
    origin_col: int
    tile_fname: str

@dataclass
class TilerClass:
    """
    This class divides the parent image into tiles which get
    packaged into TileItems

    Parameters:
        parent_image
        parent_file_name
        tile_width
        tile_height
        radius
        center
        output_dir
    """
    parent_image: np.ndarray
    parent_file_name: str
    tile_width: int
    tile_height: int
    radius: int
    center: tuple
    output_dir: str
    tile_meta_dict: dict 

    def __post_init__(self):
        """
        Set up parent image and class variables
        """ 
        self.tile_dir = self.output_dir + os.sep + self.parent_file_name
        parent_shape = np.shape(self.parent_image)
        self.parent_height = parent_shape[0]
        self.parent_width = parent_shape[1]

        # pad parent if fractional tiles
        if self.parent_width % self.tile_width > 0 or self.parent_height % self.tile_height > 0:
            self.pad_parent()  
            self.parent_height, self.parent_width = np.shape(self.parent_image)

    def pad_parent(self):
        """
        Pad parent image with zeros to fit an integer number of tiles
        """
        b_padding, t_padding = calculate_padding(self.parent_height,self.tile_height)
        l_padding, r_padding = calculate_padding(self.parent_width,self.tile_width) 
        self.tile_meta_dict['bottom_padding_value'] = b_padding
        self.tile_meta_dict['top_padding_value'] = t_padding
        self.tile_meta_dict['left_padding_value'] = l_padding
        self.tile_meta_dict['right_padding_value'] = r_padding
        self.center = ((self.parent_height+b_padding+t_padding)//2,(self.parent_width+l_padding+r_padding)//2)
        self.parent_image = np.pad(self.parent_image,((b_padding,t_padding),(l_padding,r_padding)))
    
    def save_parent_jpg(self,maxval=1500):
        """
        Convert data to byte array and save as jpg in given folder

        Parameters:
            maxval (float):     value to clip magnetograms at (Gauss)
        """
        parent_jpg = np.array(self.parent_image)
        # clip max values
        parent_jpg[np.where(parent_jpg>maxval)] = maxval
        parent_jpg[np.where(parent_jpg<-maxval)] = -maxval
        # scale between 0 and 255
        parent_jpg = ((parent_jpg+maxval)/(2*maxval)*255).astype(np.uint8)
        im = Image.fromarray(parent_jpg)
        im.save(self.output_dir+os.sep+self.parent_file_name+'.jpg')
        

    def cut_set_tiles(self,subset=False,thresh=100):
        """
        This function takes the parent image (numpy array like) and divides it up into 
        TileItems of the specified tile width and height.
        If subset keyword is True then crops a centered subset of the image of area (2*radius)**2 
        before dividing into tiles

        Parameters:
            subset (bool):   Whether to take subset or whole parent

        Raises:
            - AssertionError: If the diameter of the circle is greater than or equal to the parent image size.
        """

        if subset:
            # Assert the specified radius is less than the parent image
            diameter = self.radius * 2
            assert diameter <= self.parent_height and diameter <= self.parent_width, \
                'The diameter of the circle is too large for the parent image, please choose a smaller radius'
            
            num_row = diameter // self.tile_height
            num_col = diameter // self.tile_width

            # Find top left corner of the bounding box
            offset_x = ((num_col * self.tile_width) - diameter) // 2
            offset_y = ((num_row * self.tile_height) - diameter) // 2
            
            x_1 = self.center[0] - self.radius - offset_x
            y_1 = self.center[1] - self.radius - offset_y

        else:
            x_1 = 0
            y_1 = 0
            num_row = self.parent_height // self.tile_height
            num_col = self.parent_width // self.tile_width

        # create a folder called tiles
        os.makedirs(self.tile_dir+'/tiles', exist_ok=True)

        self.tile_item_list = []
        for row in range(num_row):
            for col in range(num_col):
                # Find the top left corner of the tile for cropping
                start_x = col * self.tile_width + x_1
                start_y = row * self.tile_height + y_1

                # Crop duplicate
                temp_image = self.parent_image[start_x:start_x+self.tile_width,start_y:start_y+self.tile_height]

                if subset & (np.max(np.abs(temp_image[:]))<thresh):
                    # don't save tile if data is below thresh (likely outside disk)
                    continue

                # Save as new tile to a folder called tiles 
                np.save(f'{self.tile_dir}/tiles/tile_{start_y}_{start_x}.npy',temp_image)
           
                # Create a TileItem
                tile_item = TileItem(self.tile_width, self.tile_height, start_y, start_x, \
                    tile_fname=f'tile_{start_y}_{start_x}.npy')
                self.tile_item_list.append(tile_item)

                
    def generate_tile_metadata(self) -> dict:
        """Generate metadata for tiles"""
        self.tile_meta_dict['file_format'] = 'npy'
        self.tile_meta_dict['number_child_tiles'] = len(self.tile_item_list)
        self.tile_meta_dict['tile_list'] = self.tile_item_list
        self.tile_meta_dict['center'] = self.center
        self.tile_meta_dict['radius'] = self.radius #assuming this is completed in other class
        return self.tile_meta_dict
    
    def convert_export_dict_to_json(self):
        """Convert metadata to json"""
        dicti = self.tile_meta_dict

        os.makedirs(self.tile_dir+'/tile_meta_data', exist_ok=True)
        with open(f'{self.tile_dir}/tile_meta_data/{self.parent_file_name}_metadata.json', 'w') as outfile:
            json.dump(dicti, outfile)
        
        return


if __name__ == '__main__':
    # these are example values set the init
    radius = int((0.9*1024)//2)
    center = (512,512)
    out_dir = 'data/test_tiles'
    tile_dim = 64

    for file in os.listdir('data/test'):
        data = h5py.File('data/test/'+file,'r')['magnetogram']
        date_time = file.split('.')[-2].strip('_TAI')
        fname = date_time+'_'+file.split('_')[0]
        file_dict = {'date_time': date_time,
                     'instrument':file.split('_')[0],
                     'resolution':1024}
        tc = TilerClass(data, fname, tile_dim, tile_dim, radius, center,out_dir,file_dict)
        tc.cut_set_tiles(subset=True)
        tc.tile_meta_dict = tc.generate_tile_metadata()
        tc.convert_export_dict_to_json()
        tc.save_parent_jpg()

