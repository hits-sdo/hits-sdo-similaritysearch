from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
import sys
sys.path.append('./')
from search_utils.image_utils import stitch_adj_imgs
import numpy as np
import torch
import glob



def sunbirdCollate(fill_type='SuperImage'):

    fill_type = fill_type

    #ret_function = sunbirdCollate('Nearest')

    #ret_function(input_tuple)

    #treat input like list..?
    def sunbird_collate(source_tuple):
        '''
        ARG:    index of dataset class
        RET:    img_t0 & img_t1   : tuple   ( both are altered img )
                label             : int
                filename          : str
        DESC: Applies random augmentation to the images then returns them for comparsion
        '''

        imgs_t0, imgs_t1, file_names, labels = [], [], [], []

        A = AugmentationList(instrument="euv")

        for item in source_tuple:
            
            file_names.append(item[2])
            labels.append(item[1])
            
            source_image = np.array(item[0]).astype(float)/255

            dict0 = A.randomize()
            dict1 = A.randomize()

            if fill_type == 'SuperImage':

                path_list = item[2].split('/')
                file_dir = path_to_data+'/'+'/'.join(path_list[:2]) +'/'
                files_list = glob.glob(file_dir+"/*.jpg", recursive = True)
                file_names = [x[len(file_dir):] for x in files_list]
                file_name = path_list[2]

                super_image = stitch_adj_imgs(data_dir=file_dir,
                                                file_name=file_name,
                                                EXISTING_FILES=file_names)

                as1 = Augmentations(super_image, dict0)
                as2 = Augmentations(super_image, dict1)

                sup_aug_img1, _ = as1.perform_augmentations()
                sup_aug_img2, _ = as2.perform_augmentations()

                # Get size of img
                img_h, img_w = 128, 128

                # Get size of super image
                sup_img_h, sup_img_w = super_image.shape[:2]

                # grab center of parent/super image
                center_sup_img_y, center_sup_img_x = (sup_img_h // 2, sup_img_w // 2)

                # grab center of tile
                center_img_y, center_img_x = (img_h // 2, img_w // 2)

                img_t0 = sup_aug_img1[
                    center_sup_img_y - center_img_y:
                    center_sup_img_y + center_img_y,
                    center_sup_img_x - center_img_x:
                    center_sup_img_x + center_img_x
                    ]

                img_t1 = sup_aug_img2[
                    center_sup_img_y - center_img_y:
                    center_sup_img_y + center_img_y,
                    center_sup_img_x - center_img_x:
                    center_sup_img_x + center_img_x
                    ]

            else:
                Aug0 = Augmentations(source_image, dict0)
                Aug1 = Augmentations(source_image, dict1)

                img_t0, _ = Aug0.perform_augmentations(fill_void=fill_type)
                img_t1, _ = Aug1.perform_augmentations(fill_void=fill_type)

            imgs_t0.append(img_t0)
            imgs_t1.append(img_t1)

        imgs_t0 = torch.from_numpy(np.array(imgs_t0)).permute(0, 3, 1, 2)
        imgs_t1 = torch.from_numpy(np.array(imgs_t1)).permute(0, 3, 1, 2)

        labels = torch.tensor(labels)

        return (imgs_t0, imgs_t1), labels, file_names
    return sunbird_collate
