import os, random, shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import os.path
import pyprojroot
root = pyprojroot.here()
utils_dir = root / 'search_utils'
import sys
sys.path.append(str(root))
from search_utils.image_utils import read_image
from search_simclr.simclr.dataloader import dataset_aug
from search_simclr.simclr.dataloader.dataset import SdoDataset, partition_tile_dir_train_val
from search_utils.file_utils import get_file_list

def main():
    print(root)
    
    example_tile_path = os.path.join(root , 'data', 'AIA211_193_171_Miniset', '20100703_000002_aia_211_193_171','tiles' , '20100703_000002_aia_211_193_171_tile_3584_1536.jpg')
    
    example_image = read_image(example_tile_path, 'jpg')
    
    ex_img_dict = {'image': example_image, 'filename' : '20100703_000002_aia_211_193_171_tile_3584_1536.jpg'}
    
    #blurr image
    blurr_img_obj = dataset_aug.Blur((5,5))
    blurr_img_dict = blurr_img_obj(ex_img_dict)
    
    #brighten image
    brighten_img_obj = dataset_aug.Brighten(0.5)
    brighten_img_dict = brighten_img_obj(ex_img_dict)
    
    #Horizontal flip
    hor_flip_obj = dataset_aug.H_Flip()
    hor_flip_img_dict = hor_flip_obj(ex_img_dict)
    
    #Vertical flip
    ver_flip_obj = dataset_aug.V_Flip()
    ver_flip_img_dict = ver_flip_obj(ex_img_dict)
    
    #Polariy flip
    p_flip_obj = dataset_aug.P_Flip()
    p_flip_img_dict = p_flip_obj(ex_img_dict)
    
    #Zoom
    zoom_obj = dataset_aug.Zoom(1.5)
    zoom_img_dict = zoom_obj(ex_img_dict)
    
    #Cutout
    cutout_obj = dataset_aug.Cutout(1, 1.0)
    cutout_img_dict = cutout_obj(ex_img_dict)
    
    #Add noise
    add_noise_obj = dataset_aug.AddNoise(0.0, 0.05)
    add_noise_img_dict = add_noise_obj(ex_img_dict)
    
    
    neighbour_tiles = ['20100703_000002_aia_211_193_171_tile_3456_1408.jpg',
                       '20100703_000002_aia_211_193_171_tile_3456_1536.jpg',
                       '20100703_000002_aia_211_193_171_tile_3456_1664.jpg',
                       '20100703_000002_aia_211_193_171_tile_3584_1408.jpg',
                       '20100703_000002_aia_211_193_171_tile_3584_1536.jpg',
                       '20100703_000002_aia_211_193_171_tile_3584_1664.jpg',
                       '20100703_000002_aia_211_193_171_tile_3712_1408.jpg',
                       '20100703_000002_aia_211_193_171_tile_3712_1536.jpg',
                       '20100703_000002_aia_211_193_171_tile_3712_1664.jpg',
                       ]
    
    
    
    #Stitch
    tile_dir = os.path.join(root , 'data')
    tot_fpath_wfname = os.path.join(root , 'data' , 'train_val_simclr', 'tot_full_path_files.txt')
    tot_file_list = neighbour_tiles#get_file_list(tot_fpath_wfname)
    #train_file_list, val_file_list = partition_tile_dir_train_val(tot_file_list, 0.8)
    data_dir=os.path.join(tile_dir, 'AIA211_193_171_Miniset', '20100703_000002_aia_211_193_171','tiles')
    file_list=tot_file_list
    stitch_obj = dataset_aug.StitchAdjacentImagesVer2(data_dir, file_list)
    stitch_img_dict = stitch_obj(ex_img_dict)
    
    
    #Crop
    crop_obj = dataset_aug.Crop()
    crop_img_dict = crop_obj(ex_img_dict)
    
    #Rotate
    rotate_obj = dataset_aug.Rotate(90.0)
    rotate_img_dict = rotate_obj(stitch_img_dict)
    rotate_img_dict = crop_obj(rotate_img_dict)
    
    #Translate
    translate_obj = dataset_aug.Translate((50,0))
    translate_img_dict = translate_obj(stitch_img_dict)
    translate_img_dict = crop_obj(translate_img_dict)
    
    
    
    
    
    # pltsubplot(1,2,1)
    # plt.imshow(blurr_img_dict['image'])
    
    # Plot images side-by-side
    # plt.imshow(ex_img_dict['image'][:, :, 0], cmap='hot')
    # plt.title("original image")
    # # plt.imshow(test_image.squeeze())
    # plt.show()

    plt.subplot(2, 5, 1)
    plt.imshow(blurr_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("blur")
    
    plt.subplot(2, 5, 2)
    plt.imshow(add_noise_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("noise")
    
    plt.subplot(2, 5, 3)
    plt.imshow(brighten_img_dict['image'][:, :, 0], cmap='hot', vmax=ex_img_dict['image'].max(), vmin=ex_img_dict['image'].min())
    plt.title("brighten")
    
    plt.subplot(2, 5, 4)
    plt.imshow(hor_flip_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("horizontal flip")
    
    plt.subplot(2, 5, 5)
    plt.imshow(ver_flip_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("vertical flip")
    
    plt.subplot(2, 5, 6)
    plt.imshow(p_flip_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("polarity flip")
    
    plt.subplot(2, 5, 7)
    plt.imshow(rotate_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("rotate")
    
    plt.subplot(2, 5, 8)
    plt.imshow(translate_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("translate")
    
    plt.subplot(2, 5, 9)
    plt.imshow(zoom_img_dict['image'][:, :, 0], cmap='hot')
    plt.title("zoom")
    
    plt.subplot(2, 5, 10)
    plt.imshow(cutout_img_dict['image'][:, :, 0], cmap='hot')#, vmax=ex_img_dict['image'].max(), vmin=ex_img_dict['image'].min())
    plt.title("cutout")
    
    # plt.subplot(2, 5, 12)
    # plt.imshow(stitch_img_dict['image'][:, :, 0], cmap='hot')
    # plt.title("stitch")
    
    # plt.subplot(2, 5, 13)
    # plt.imshow(crop_img_dict['image'][:, :, 0], cmap='hot')
    # plt.title("crop")
    
    #plt.tight_layout()
    
    plt.show()
    
    print(ex_img_dict['image'].shape)
    
    # #create the 2D array of standard distribution
    # data = [[-100,2,3], [4,5,6], [7,8,9]]
    # f = np.array(data)
    # print(f[1,2])
    # # 6
    # print(data[1][2])
    # # 6

    # plt.imshow(f, cmap='Greys', interpolation="nearest", origin="upper")
    # plt.colorbar()
    # plt.show()
    
    
# class Transforms_SimCLR(object):
#     def __init__(self, 
#                  blur, 
#                  brighten, 
#                  translate, 
#                  zoom, 
#                  rotate, 
#                  noise_mean, 
#                  noise_std, 
#                 cutout_holes, 
#                 cutout_size, 
#                 data_dir,
#                 file_list
#                 ):
#         #print(translate)
        
#         self.train_transform = transforms.Compose([
#             # Stitch image should happen before the fill voids
#             StitchAdjacentImagesVer2(data_dir, file_list),
#             transforms.RandomApply([H_Flip()], p=0.5),
#             transforms.RandomApply([V_Flip()], p=0.5),
#             transforms.RandomApply([P_Flip()], p=0.5), 
#             transforms.RandomApply([Rotate(rotate)], p=0.5),
#             transforms.RandomApply([Brighten(brighten)], p=0.5),
#             transforms.RandomApply([Translate(translate)], p=0.5),
#             transforms.RandomApply([Zoom(zoom)], p=0.5),
#             transforms.RandomApply([Cutout(cutout_holes, cutout_size)], p=0.5),
#             transforms.RandomApply([Blur(blur)], p=0.5),
#             transforms.RandomApply([AddNoise(noise_mean, noise_std)], p=0.5),
#             Crop(),
#         ToTensor()])
        
        

#         self.test_transform = transforms.ToTensor()
    
#     def __call__(self, sample):
#         transformed_sample1 = self.train_transform(sample)
#         transformed_sample2 = self.train_transform(sample)
#         return transformed_sample1, transformed_sample2
    
    
    
if __name__== "__main__":
    main()