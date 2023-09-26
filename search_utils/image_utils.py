import pickle
from PIL import Image
import numpy as np
import os

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

def interpolate_superimage(img, loc):
    image_top_left = (img.shape[0]*loc[0]//3, img.shape[1]*loc[1]//3)
    image_bottom_right = (img.shape[0]*(loc[0] +1)//3, img.shape[1]*(loc[1] +1)//3)
    tile_y, tile_x = img.shape[0]//3, img.shape[1]//3

    y_lower = 0 if image_top_left[0] - 1 < 0 else image_top_left[0] - 1
    x_lower = 0 if image_top_left[1] - 1 < 0 else image_top_left[1] - 1
    y_upper = img.shape[0] - 1 if image_bottom_right[0] == img.shape[0] else image_bottom_right[0] + 1
    x_upper = img.shape[1] - 1 if image_bottom_right[1] == img.shape[1] else image_bottom_right[1] + 1

    m = np.median(img[tile_y:2*tile_y, tile_x:2*tile_x],
                  axis=(0, 1))*np.ones((tile_y, tile_x, 3))
    n = 1.0
    i = np.linspace(0, tile_y-1, tile_y)
    j = np.linspace(0, tile_x-1, tile_x)
    x, y = np.meshgrid(j, i)
    x = image_top_left[1] + x
    y = image_top_left[0] + y

    d1 = 1/(0.001 + y - y_lower)**n
    d2 = 1/(0.001 + y_upper - y)**n
    d3 = 1/(0.001 + x - x_lower)**n
    d4 = 1/(0.001 + x_upper - x)**n

    d1 = np.repeat(d1[:, :, np.newaxis], 3, axis=2)
    d2 = np.repeat(d2[:, :, np.newaxis], 3, axis=2)
    d3 = np.repeat(d3[:, :, np.newaxis], 3, axis=2)
    d4 = np.repeat(d4[:, :, np.newaxis], 3, axis=2)

    if y_lower == 0:
        I1 = m
    else:
        I1 = np.zeros((tile_y, tile_x, 3))
        for k in range(3):
            I1[:, :, k] = np.repeat(img[y_lower, x[0, :].astype('int'), k].reshape(1, tile_x), repeats=tile_y, axis=0)
    
    if y_upper == img.shape[0] - 1:
        I2 = m
    else:
        I2 = np.zeros((tile_y, tile_x, 3))
        for k in range(3):
            I2[:, :, k] = np.repeat(img[y_upper, x[0, :].astype('int'), k].reshape(1, tile_x), repeats=tile_y, axis=0)

    if x_lower == 0:
        I3 = m
    else:
        I3 = np.zeros((tile_y, tile_x, 3))
        for k in range(3):
            I3[:, :, k] = np.repeat(img[y[:, 0].astype('int'), x_lower, k].reshape(1, tile_y), repeats=tile_x, axis=0).T
    
    if x_upper == img.shape[1] - 1:
        I4 = m
    else:
        I4 = np.zeros((tile_y, tile_x, 3))
        for k in range(3):
            I4[:, :, k] = np.repeat(img[y[:, 0].astype('int'), x_upper, k].reshape(1, tile_y), repeats=tile_x, axis=0).T

    img[image_top_left[0]:image_top_left[0]+tile_y,
        image_top_left[1]:image_top_left[1]+tile_x, :] = (I1*d1 + I2*d2 + I3*d3 + I4*d4)/(d1 + d2 + d3 + d4)
    
    return img
    


def stitch_adj_imgs(data_dir, file_name,
                    EXISTING_FILES,
                    multi_wl=False,
                    iterative=False,
                    remove_coords=False):
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
        multi_wl (bin):
            set as True when multi-wavelength images are used
        iterative (bin):
            set as True to iteratively fill blank spaces in superimage
        remove_coords (bin):
            set as True to artificially create blank space in superimage


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

    coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    if remove_coords is True:
        removed_coords = [(1, 0), (0, 1), (0, 2)]
        for i in removed_coords:
            coordinates.remove(i)
    else:
        removed_coords = []

    source_image = read_image(os.path.join(data_dir, file_name), file_format)

    image_len = source_image.shape[0]
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
            im = read_image(os.path.join(data_dir, tile_name), file_format)
            superImage[i*image_len: (i+1)*image_len, j*image_len:
                       (j+1)*image_len] = im
        else:
            removed_coords.append((i, j))

    # print(len(removed_coords))
    
    if len(removed_coords)>0 and iterative==True:
        l = []
        for k in range(20):     
            for loc in removed_coords:               
                superImage = interpolate_superimage(superImage, loc)

    return superImage

