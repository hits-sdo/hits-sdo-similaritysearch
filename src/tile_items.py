"""TileItem utilizes NamedTuple"""
import numpy as np
import os
import json
# import pyprojroot
from dataclasses import dataclass
from search_utils.image_utils import *
from typing import NamedTuple 
from PIL import Image

# base_path = pyprojroot.find_root(pyprojroot.has_dir('.git'))

# {parent_dim: (w,h), parent_padded: (w,h), tile_width....}
class TileItem(NamedTuple):
    tile_width:int
    tile_height:int
    origin_row: int
    origin_col: int
    tile_fname: str

#populate with functions in TilerClass

tempDict = {
    'instrument': 'AIA',
    'date': 'APRIL 4, 2021',
    'time': '12:00:00',
    'wavelength': '193.4823420',
    'AIA_or_HMI': 'AIA',
    'padding': '(2,34,5)',
    'number_child_tiles': '100',
    'tile_list': (),
    'center': (0,0),
    'radius': 5
}

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
        self.output_dir = self.output_dir + os.sep + self.parent_file_name
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
    
    def cut_set_tiles(self,subset=False):
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

            # Find the number of tiles to build up vertically and across horizontally
            diameter_bound_x, num_col = 0, 0
            diameter_bound_y, num_row = 0, 0

            while (diameter_bound_x < diameter) and (diameter_bound_x + self.tile_width <= self.parent_width):
                diameter_bound_x += self.tile_width
                num_col += 1

            while (diameter_bound_y < diameter) and (diameter_bound_y + self.tile_height <= self.parent_height):
                diameter_bound_y += self.tile_height
                num_row += 1

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
        os.makedirs(self.output_dir+'/tiles', exist_ok=True)

        self.tile_item_list = []
        for row in range(num_row):
            for col in range(num_col):
                # Find the top left corner and width and height of the tile for cropping
                start_x = col * self.tile_width + x_1
                start_y = row * self.tile_height + y_1
                width = self.tile_width + start_x
                height = self.tile_height + start_y

                # Crop duplicate
                temp_image = self.parent_image[start_x:start_x+width,start_y:start_y+height]

                # Save as new tile to a folder called tiles 
                np.save(f'{self.output_dir}/tiles/tile_{start_y}_{start_x}.npy',temp_image)
           
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

        os.makedirs(self.output_dir+'/tile_meta_data', exist_ok=True)
        with open(f'{self.output_dir}/tile_meta_data/{self.parent_file_name}_metadata.json', 'w') as outfile:
            json.dump(dicti, outfile)
        
        return

    # def reconstruct_parent_img(self):
    #     """Reconstruct parent image from tiles"""
        
    #     pass


    


if __name__ == '__main__':
    #parent_height = ParentTransformationsFromTile.parent_img_height_after_padding
    #parent_width = ParentTransformationsFromTile.parent_img_width_after_padding

    #these are example values set the init
    # parent_height = 4096
    # parent_width = 4096
    # dicti = generate_tile_metadata()
    cx = 4096//2
    cy = 4096//2
    tc = TilerClass(None, 512, 1024, '', 4096, 4096, 'data/raw/latest_4096_0193.jpg',
        tempDict, '', [], 1792, (cx,cy), '', '')

    tc.generate_tile_fpath_write()
    tc.cut_subset_tiles()
    # tc.cut_set_tiles()
    tc.tile_meta_dict = tc.generate_tile_metadata()
    tc.convert_export_dict_to_json()

    # parent_image: bytearray
    # tile_width: int
    # tile_height: int
    # tile_path_output: str
    # parent_height : int
    # parent_width : int
    # parent_path_input : str
    # tile_meta_dict: dict
    # tile_meta_dict_path: str
    # tile_item_list: list[TileItem]
    # radius: int
    # center: tuple
    # output_dir: str
