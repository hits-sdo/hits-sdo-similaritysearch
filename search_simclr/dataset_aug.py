from torchvision import transforms
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import search_utils.augmentation_list
from search_utils.augmentation_list import AugmentationList
from search_utils.augmentations import Augmentations
# import cv2 as cv


# rewrite all functions as classes and put those inside call
# because then we can use that 
# list of augmentations (blur, brighten, h_flip, p_flip, v_flip, rotate, translate, zoom) 
# respect ranges for augmentations: 
    #     self.zoom_range = (0.8, 1.2)
    #     self.brighten_range = (0.5, 1.5)
    #     self.rotate_range = (-180, 180)
    #     # self.blur_range = ((1, 1), (2, 2))
    #     self.translate_range = (-10, 10)

#TODO: let's figure out how to get our augmentations class imported (local probably?)

def random_augment_image(img):
    # Make random augmentation dictionary
    augment_list = AugmentationList(instument = "euv") # or mag
    rand_dict = augment_list.randomize()
        
    # Preform Augmentations
    augments = Augmentations(img, rand_dict)
    augmented_img, title = augments.perform_augmentations()
    return augmented_img
    
class Blur(object):
    def __init__(self, value):
        assert isinstance(value, (int, int))
        
        
    def __call__(self, img, value):
        """
        Blurs the image by the amount by blur (default = (1, 1))
        Blurring is performed as an average blurring
        with kernel size defined by blur
    """
        image = cv.blur(img, (value[0], value[1]), 0)
        return image
    