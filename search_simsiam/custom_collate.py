from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations
import numpy as np
import torch


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

    fill_type = 'Nearest'

    for item in source_tuple:
        file_names.append(item[2])
        labels.append(item[1])
        source_image = np.array(item[0]).astype(float)/255
        dict0 = A.randomize()
        dict1 = A.randomize()
        Aug0 = Augmentations(source_image, dict0)
        Aug1 = Augmentations(source_image, dict1)
        img_t0, _ = Aug0.perform_augmentations(fill_void=fill_type)
        img_t1, _ = Aug1.perform_augmentations(fill_void=fill_type)
        imgs_t0.append(img_t0)
        imgs_t1.append(img_t1)

    imgs_t0 = torch.from_numpy(np.array(imgs_t0)).permute(0,3,1,2)
    imgs_t1 = torch.from_numpy(np.array(imgs_t1)).permute(0,3,1,2)
    labels = torch.tensor(labels)
    return (imgs_t0, imgs_t1), labels, file_names
