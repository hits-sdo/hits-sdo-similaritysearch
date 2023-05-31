from torchvision import transforms
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import search_utils.augmentation_list
from search_utils.augmentation_list import AugmentationList
from search_utils.augmentations import Augmentations
import cv2 as cv
import numpy as np

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
    augment_list = AugmentationList(instrument = "euv")  # or mag
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
    
class H_Flip(object):
    def __init__(self, value):
        assert isinstance(value, (bool))
        
    def __call__(self, img, value):
        return cv.flip(img, 1)
        
class V_Flip(object):
    def __init__(self, value):
        assert isinstance(value, (bool))
        
    def __call__(self, img, value):
        return cv.flip(img, 0)
    
class P_Flip(object):
    def __init__(self, value):
        assert isinstance(value, (bool))
        
    def __call__(self, img, value):
        return (1-img)
    
class Brighten(object):
    def __init__(self, value) -> None:
        assert isinstance(value, float)

    def __call__(self, img, value):
        return np.abs(img)**value
    
class Translate(object):
    def __init__(self, value):
        assert isinstance(value, (int, int))
    
    def __call__(self, img, value):
        s = img.shape
        m = np.float32([[1, 0, value[0]], [0, 1, value[1]]])
        # Affine transformation to translate the image and output size
        img = cv.warpAffine(img, m, (s[1], s[0]))
        return img
    
class Zoom(object):
    def __init__(self, value):
        assert isinstance(value, float)
    
    def __call__(self, img, value):
        s = img.shape
        s1 = (int(value*s[0]), int(value*s[1]))
        img_zeros = np.zeros(s)

        image_resize = cv.resize(img, (s1[1], s1[0]), interpolation=cv.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if value < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img_zeros[y1:y2, x1:x2] = image_resize
            return img_zeros
        else:
            return image_resize

class Rotate(object):
    def __init__(self, value):
        assert isinstance(value, float)

    def __call__(self, img, value):
        s = img.shape
        cy = (s[0]-1)/2  # y center : float
        cx = (s[1]-1)/2  # x center : float
        M = cv.getRotationMatrix2D((cx, cy), value, 1)  # rotation matrix
    
        # Affine transformation to rotate the image and output size s[1],s[0]
        return cv.warpAffine(img, M, (s[1], s[0]))
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)   