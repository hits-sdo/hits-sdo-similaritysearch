from torchvision import transforms
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import search_utils.augmentation_list
from search_utils.augmentation_list import AugmentationList
from search_utils.augmentation import Augmentations
import cv2 as cv
import numpy as np
import torch

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
        self.blur = value
        assert isinstance(value, tuple)
        assert isinstance(value[0], int)
        assert isinstance(value[1], int)      
        
    def __call__(self, img):
        """
        Blurs the image by the amount by blur (default = (1, 1))
        Blurring is performed as an average blurring
        with kernel size defined by blur
    """
        image = cv.blur(img, (self.blur[0], self.blur[1]), 0)
        return image
    
class H_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, img):
        return cv.flip(img, 1)
        
class V_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, img):
        return cv.flip(img, 0)
    
class P_Flip(object):
    def __init__(self):
        ...
        
    def __call__(self, img):
        return (1-img)
    
class Brighten(object):
    def __init__(self, value) -> None:
        self.brighten = value
        assert isinstance(value, float)

    def __call__(self, img):
        return np.abs(img)**self.brighten
    
class Translate(object):
    def __init__(self, value):
        self.translate = value
        assert isinstance(value, tuple)
        assert isinstance(value[0], int)
        assert isinstance(value[1], int)

    
    def __call__(self, img):
        s = img.shape
        m = np.float32([[1, 0, self.translate[0]], [0, 1, self.translate[1]]])
        # Affine transformation to translate the image and output size
        img = cv.warpAffine(img, m, (s[1], s[0]))
        return img
    
class Zoom(object):
    def __init__(self, value):
        self.zoom = value
        assert isinstance(value, float)
    
    def __call__(self, img):
        s = img.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img_zeros = np.zeros(s)

        image_resize = cv.resize(img, (s1[1], s1[0]), interpolation=cv.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
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
        self.rotate = value
        assert isinstance(value, float)

    def __call__(self, img):
        s = img.shape
        cy = (s[0]-1)/2  # y center : float
        cx = (s[1]-1)/2  # x center : float
        M = cv.getRotationMatrix2D((cx, cy), self.rotate, 1)  # rotation matrix
    
        # Affine transformation to rotate the image and output size s[1],s[0]
        return cv.warpAffine(img, M, (s[1], s[0]))
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        print(f'Image Shape = {img.shape}')
        
        # If the image is grayscale, expand its dimensions to have a third axis
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)


class Transforms_SimCLR(object):
    def __init__(self, blur, brighten, translate, zoom, rotate):
        print(translate)
        self.train_transform = transforms.Compose([
            transforms.RandomApply([H_Flip()], p=0.5),
            transforms.RandomApply([V_Flip()], p=0.5),
            transforms.RandomApply([P_Flip()], p=0.5), 
            transforms.RandomApply([Rotate(rotate)], p=0.5),
            transforms.RandomApply([Brighten(brighten)], p=0.5),
            transforms.RandomApply([Translate(translate)], p=0.5),
            transforms.RandomApply([Zoom(zoom)], p=0.5),
            transforms.RandomApply([Blur(blur)], p=0.5),
        ToTensor()])

        self.test_transform = transforms.ToTensor()
    
    def __call__(self, img):
        return self.train_transform(img), self.train_transform(img)
    

    
    #transforms = transforms.RandomApply(torch.nn.ModuleList([
    #transforms.ColorJitter(),
    # ]), p=0.3)
    
def main():
    ...
    
if __name__ == "__main__":
    main()