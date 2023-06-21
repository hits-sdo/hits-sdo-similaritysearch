import pickle
from PIL import Image
import numpy as np
from lightly.data import LightlyDataset
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sdo_augmentation.augmentation import Augmentations
import cv2


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

    y_lower = 0 if image_top_left[0]-1<0 else image_top_left[0] - 1
    x_lower = 0 if image_top_left[1]-1<0 else image_top_left[1] - 1
    y_upper = img.shape[0] -1 if image_bottom_right[0]==img.shape[0] else image_bottom_right[0] + 1
    x_upper = img.shape[1] -1 if image_bottom_right[1]==img.shape[1] else image_bottom_right[1] + 1
    m = np.median(img[128:256,128:256],axis=(0,1))
    n = 1.0
    for i in range(img.shape[0]//3):
        for j in range(img.shape[1]//3):
            y = image_top_left[0] + i
            x = image_top_left[1] + j
            # d1 = (y_upper - y)
            # d2 = (y - y_lower)
            # d3 = (x_upper - x)
            # d4 = (x - x_lower)
            d1 = 1/(0.001 + y - y_lower)**n
            d2 = 1/(0.001 + y_upper - y)**n
            d3 = 1/(0.001 + x - x_lower)**n
            d4 = 1/(0.001 + x_upper - x)**n
            I1 = img[y_lower, x] if y_lower != 0 else m
            I2 = img[y_upper, x] if y_upper != img.shape[0] -1 else m 
            I3 = img[y, x_lower] if x_lower != 0 else m
            I4 = img[y, x_upper] if x_upper != img.shape[1] -1 else m

            img[y,x] = (I1*d1 + I2*d2 + I3*d3 + I4*d4)/(d1 + d2 + d3 + d4)
    
    return img
    


def stitch_adj_imgs(data_dir, file_name, EXISTING_FILES, multi_wl=False, iterative = False):
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

    # coordinates = [(1,1)]
    # removed_coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

    # coordinates = [(0, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    # removed_coords = [(1,0), (0, 1), (0, 2)]

    coordinates = [(0, 0), (0, 1), (0, 2), (1,0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    source_image = read_image(data_dir + file_name, file_format)
    image_len = source_image.shape[0]
    superImage = np.zeros((3*image_len, 3*image_len, 3))
    # superImage = np.pad(source_image, ((image_len, image_len),
    #                                    (image_len, image_len), (0, 0)),
    #                                    'edge')

    removed_coords = []

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
        else:
            removed_coords.append((i, j))
    
    if len(removed_coords)>0 and iterative==True:
        l = []
        for k in range(20):
            s0 = superImage.copy()
            count = 0      
            for loc in removed_coords:
                plt.imshow(superImage)
                plt.title(f'iteration {k+1}')
                plt.savefig('/home/schatterjee/Desktop/superImage_animation/padding_'+str(k).zfill(2)+'_'+
                            str(count)+'.png')
                # y0 = image_len*loc[0]
                # y1 = image_len*(1+loc[0])
                # x0 = image_len*loc[1]
                # x1 = image_len*(1+loc[1])
                # blurred = cv2.blur(superImage,(32,32))
                # superImage[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
                
                superImage = interpolate_superimage(superImage, loc)
                count += 1
            s1 = superImage
            delta = 100*np.sum((s1-s0)**2)/np.sum(s0**2)
            l.append(delta)
            if delta<10**-4:
                break


    return superImage

if __name__ == '__main__':
    idx = 10
    path_to_data = '/home/schatterjee/Documents/hits/aia_171_color_1perMonth'
    dataset = LightlyDataset(input_dir=path_to_data)
    item = dataset[idx]
    print(item)
    source_image = np.array(item[0]).astype(float)/255
    path_list = item[2].split('/')
    file_dir = path_to_data+'/'+'/'.join(path_list[:2]) +'/'
    files_list = glob.glob(file_dir+"/*.jpg", recursive = True)
    file_names = [x[len(file_dir):] for x in files_list]
    file_name = path_list[2]

    super_image = stitch_adj_imgs(data_dir=file_dir,
                                  file_name=file_name,
                                  EXISTING_FILES=file_names,
                                  multi_wl = False)
    
    a = Augmentations(super_image,dct={'translate':(30,30),'rotate':15})
    img_t, _ = a.perform_augmentations()
    plt.subplot(1,3,1)
    plt.imshow(source_image)
    plt.title('Original')
    plt.subplot(1,3,2)
    plt.imshow(super_image)
    plt.gca().add_patch(Rectangle((128,128),128,128,
                    edgecolor='green',
                    facecolor='none',
                    lw=2))
    plt.title('Padding 7')
    plt.subplot(1,3,3)
    plt.imshow(img_t[128:256,128:256])
    plt.title('Augmented')

    # plt.plot(l)
    # plt.yscale('log')
    # print(np.max(l))
    plt.show()