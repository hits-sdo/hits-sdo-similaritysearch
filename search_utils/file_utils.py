import os
import pyprojroot

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