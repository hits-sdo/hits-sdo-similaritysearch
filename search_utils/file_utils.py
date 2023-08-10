import os
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))
from search_simclr.simclr.dataloader.dataset import partition_tile_dir_train_val

"""
- [ ] `search_utils/file_utils.py` Add utility functions to transverse new file tree of jpgs from `aia_171_color_1perMonth.tar.gz` 
    - [ ] `get_file_list()`
    - [ ] `get_file_list_from_dir()`
    - [ ] `get_file_list_from_dir_recursive()`
"""

def get_file_list(file_list_txt: str) -> list:
    ''' 
    Takes a file name as a string object and returns a list of 
    strings associated with image files.
    
    Args: 
    file_list_txt(str): text file with a list of file names, 
    each file on a new line
    
    Return:
    tile_list(list): list of strings associated with fil names
    from file_list_txt
    '''
    tile_list = list[str]
    with open(file_list_txt, 'r') as file:
        tile_list = [line.strip() for line in file.readlines()]

    return tile_list


# Splits the total file list into train and val lists
def split_val_files(tot_txt_path, train_file_list_txt_path, val_file_list_txt_path, num_imgs=None, percent_split=0.8):
    tot_file_list = get_file_list(tot_txt_path)
    
    # Partition the data
    if num_imgs is not None: 
        train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list, percent_split)
    else:  
        train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list[:num_imgs], percent_split)

    # Write to files
    with open(os.path.join(train_file_list_txt_path), 'w') as f:
        for item in train_file_list:
            f.write("%s\n" % item)
    with open(os.path.join(val_file_list_txt_path), 'w') as f:
        for item in val_file_list:
            f.write("%s\n" % item)
            
    return train_file_list, val_file_list