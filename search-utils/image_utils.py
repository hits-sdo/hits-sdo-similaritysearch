import pickle
from PIL import Image
import numpy as np


def read_image(image_loc, image_format):
    """
    read images in pickle/jpg/png format
    and return normalized numpy array
    """

    if (image_format == 'p'):
        image = pickle.load(imfile := open(image_loc, 'rb'))
        imfile.close()
        image = image.astype(float) / 255

    if (image_format == 'jpg' or image_format == 'png'
            or image_format == 'jpeg'):
        im = Image.open(image_loc)
        image = np.array(im).astype(float) / 255

    return image


def stitch_adj_imgs(data_dir, file_name, EXISTING_FILES):
    """
    stitches adjacent images to return a superimage
    """
    len_ = len(file_name)-len('0000_0000.p')
    iStart = int(file_name[-11:-7])
    jStart = int(file_name[-6:-2])
    # coordinates of surrounding tiles
    coordinates = [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    image_len = read_image(data_dir + file_name, 'p').shape[0]
    superImage = np.zeros((3*image_len, 3*image_len))
    for i, j in coordinates:
        i_s = iStart - image_len + i * image_len
        j_s = jStart - image_len + j * image_len

        tile_name = \
            f"{file_name[0:len_]}{str(i_s).zfill(4)}_{str(j_s).zfill(4)}.p"
        if tile_name in EXISTING_FILES:
            im = read_image(data_dir + tile_name, 'p')
            superImage[i*image_len: (i+1)*image_len, j*image_len:
                       (j+1)*image_len] = im

    return superImage
