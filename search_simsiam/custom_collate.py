from sdo_augmentation.augmentation_list import AugmentationList
from sdo_augmentation.augmentation import Augmentations

def sunbird_collate(source_tuple):
    '''
    ARG:    index of dataset class
    RET:    img_t0 & img_t1   : tuple   ( both are altered img )
            label             : int
            filename          : str
    DESC: Applies random augmentation to the images then returns them for comparsion
    '''
    label = source_tuple[1]
    file_name = source_tuple[2]

    source_image = np.array(source_tuple[0])
    
    print(source_image.shape)

    A = AugmentationList(instrument="euv")
    dict0 = A.randomize()
    dict1 = A.randomize()
    
    if 'brighten' in list(dict0.keys()):
      del dict0['brighten']

    if 'brighten' in list(dict1.keys()):
      del dict1['brighten']

    print(dict0, "\n", dict1)

    Aug0 = Augmentations(source_image, dict0)
    Aug1 = Augmentations(source_image, dict1)


    fill_type = None
    img_t0, _ = Aug0.perform_augmentations(fill_void=fill_type)
    img_t1, _ = Aug1.perform_augmentations(fill_void=fill_type)
    
    # t0 = torch.from_numpy(img_t0)
    # t1 = torch.from_numpy(img_t1)



    return (img_t0,img_t1),label,file_name