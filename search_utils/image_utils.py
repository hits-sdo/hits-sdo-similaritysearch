import pickle
from PIL import Image
import numpy as np


def read_image(image_loc, image_format):
    '''
    reads in image

    Parameters:
        image_loc(str):
            path (includes filename)
        image_format(str):
            .jpg or .p or .png

    Returns:
        image(np array)    
            read-in image
    '''

    if (image_format == 'p'):
        image = pickle.load(imfile := open(image_loc, 'rb'))
        imfile.close()
        image = image.astype(float) / 255

    if (image_format == 'jpg' or image_format == 'png'
            or image_format == 'jpeg'):
        im = Image.open(image_loc)
        image = np.array(im).astype(float) / 255

    return image

def stitch_adj_imgs(data_dir, file_name, EXISTING_FILES, multi_wl=False):
    '''
    Stitches surrounding 8 images to inputted image in 
    order to not have unfilled edges 

    Parameters:
        data_dir(str):
            directory of images
        file_name(str):
            file name of image to find neighbors too (?)
        EXISTING_FILES(str):
            list of filenames inside data_dir

    Returns:
        super_Image(np array)    
            stitched parent image

    Ex file_name Format:
        multi_wl = False
        '20100601_000036_aia.lev1_euv_12s_4k_tile_2688_768.jpg'
        multi_wl = True
        '20100601_000008_aia_211_193_171_tile_1024_2304.jpg'
    '''

    if multi_wl is False:
        date_instrument, tile_info, file_format = file_name.split('.')
    else:
        tile_info, file_format = file_name.split('.')
        tile_info_list = tile_info.split('_')
        idx = tile_info_list.index('aia') + 1
        date_instrument = '_'.join(tile_info_list[:idx])
        tile_info = '_'.join(tile_info_list[idx:])
    
    list_info = tile_info.split('_')

    iStart, jStart = np.array(list_info[-2:]).astype('int')

    list_info_constant = '_'.join(list_info[:-2])

    coordinates = [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    image_len = read_image(data_dir + file_name, file_format).shape[0]
    superImage = np.zeros((3*image_len, 3*image_len, 3))
    for i, j in coordinates:
        i_s = iStart - image_len + i * image_len
        j_s = jStart - image_len + j * image_len

        tile_info = "_".join([list_info_constant, str(i_s), str(j_s)])

        if multi_wl is False:
            tile_name = ".".join([date_instrument, tile_info, file_format])
        else:
            tile_name = ".".join(['_'.join([date_instrument, tile_info]),
                                  file_format])
                   
        if tile_name in EXISTING_FILES:
            im = read_image(data_dir + tile_name, file_format)
            superImage[i*image_len: (i+1)*image_len, j*image_len:
                       (j+1)*image_len] = im

    return superImage
