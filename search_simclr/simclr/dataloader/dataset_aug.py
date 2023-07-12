from typing import Any
from torchvision import transforms
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import search_utils.augmentation_list
from search_utils.augmentation_list import AugmentationList
from search_utils.augmentation import Augmentations
from search_utils.image_utils import read_image, interpolate_superimage, stitch_adj_imgs
import cv2 as cv
import numpy as np
import torch
import random
import os.path

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

    
class Blur(object):
    def __init__(self, value):
        self.blur = value
        assert isinstance(value, tuple)
        assert isinstance(value[0], int)
        assert isinstance(value[1], int)      
        
    def __call__(self, sample):
        """
        Blurs the image by the amount by blur (default = (1, 1))
        Blurring is performed as an average blurring
        with kernel size defined by blur
    """
        image = cv.blur(sample["image"], (self.blur[0], self.blur[1]), 0)
        return image

class AddNoise(object):
    """
    Adds random noise to original image as np.ndarray
    Args:
      value(float): percentage value for amount of noise added  
    """ 
    # def __init__(self, value):
    #     assert isinstance(value, float)
    #     self.noise = value

    # def __call__(self, img):
    #     noisy_image = np.copy(img)
    #     print(noisy_image.shape)
    #     height, width = noisy_image.shape #add a _ for potential third value (channel)
    #     num_pixels = int(self.noise * height * width)

    #     # Generate random pixel coordinates
    #     random_coords = np.random.randint(0, min(height, width), size=(num_pixels, 2))

    #     # Add random noise to pixels
    #     for coord in random_coords:
    #         y, x = coord
    #         r_noise = np.random.randint(0, 256, size=1)  # Generate random noise for RGB channels
    #         noisy_image[y, x] += (r_noise - 128)

    #     return noisy_image
    
    
    def __init__(self, mean=0, std_lim=0.05):
        self.mean = mean
        self.std_lim = std_lim
        

    def __call__(self, sample):
        img = sample["image"]
        self.std = random.uniform(0, self.std_lim)
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 1)
        return img
    
# class FillVoids(object):
#     """
#         Do this after stitch adj img

#         FIX THESE:
#         [ ] Adapt to three channel images
#         [ ] Change interpolation technique (to include stitched imgs as part of interpolation)
#     """
#     def __init__(self):
#         ...

#     def __call__(self, image):
#          # Fill voids
#         #image = image_utils.read_image(image_fullpath, 'p')
#         # v, h = image.shape[0]//2, image.shape[1]//2
#         # if len(image.shape) == 3:
#         #     image = np.pad(image, ((v, v), (h, h), (0, 0)), 'edge')
#         # else:
#         #     image = np.pad(image, ((v, v), (h, h)), 'edge')
            
#         # print("Image: "+image)
#         mask = cv.inRange(image, (0, 0, 0), (0, 0, 0))
#         radius = 3 # The radius around a pixel to inpaint, smaller values are less blurry
#         filled_image = cv.inpaint(image, mask, radius, cv.INPAINT_NS) # cv2.INPAINT_NS or cv2.INPAINT_TELEA

#         return filled_image
        
class StitchAdjacentImagesVer2(object): 
    
    """ 
        do stitch adj images before fill voids (Fill voids uses center tile 
        interpolation and you would want to take average of all tiles if 
        that info is available)
    """

    def __init__(self, data_dir, file_name, file_list) -> None:
        self.data_dir = data_dir
        #self.file_name = file_name
        self.file_list = file_list

    def __call__(self, sample):
        
        """
        stitches adjacent images to return a superimage
        """
        superImage = stitch_adj_imgs(self.data_dir, 
                        #self.file_name,
                        sample["filename"],
                        EXISTING_FILES=self.file_list,
                        multi_wl=False,
                        iterative=False,
                        remove_coords=False)
       

        return superImage

# class Cutout(object):
#     def __init__(self, n_holes, length):
#         assert isinstance(n_holes, int)
#         assert isinstance(length, int)
#         self.n_holes = n_holes
#         self.length = length
        

#     def __call__(self, img):
#         h, w = img.shape[:2]
#         mask = np.ones((h, w), np.float32)

#         for n in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)

#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)

#             mask[y1:y2, x1:x2] = 0

#         img = img * mask[..., np.newaxis]
#         return img

class H_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        return cv.flip(sample["image"], 1)
        
class V_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        return cv.flip(sample["image"], 0)
    
class P_Flip(object):
    def __init__(self):
        ...
        
    def __call__(self, sample):
        return (1-sample["image"])
    
class Brighten(object):
    def __init__(self, value) -> None:
        self.brighten = value
        assert isinstance(value, float)

    def __call__(self, sample):
        return np.abs(sample["image"])**self.brighten
    
class Translate(object):
    def __init__(self, value):
        self.translate = value
        assert isinstance(value, tuple)
        assert isinstance(value[0], int)
        assert isinstance(value[1], int)

    
    def __call__(self, sample):
        img = sample["image"]
        s = img.shape
        m = np.float32([[1, 0, self.translate[0]], [0, 1, self.translate[1]]])
        # Affine transformation to translate the image and output size
        img = cv.warpAffine(img, m, (s[1], s[0]))
        return img
    
class Zoom(object):
    def __init__(self, value):
        self.zoom = value
        assert isinstance(value, float)
    
    def __call__(self, sample):
        img = sample["image"]
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

    def __call__(self, sample):
        img = sample["image"]
        s = img.shape
        cy = (s[0]-1)/2  # y center : float
        cx = (s[1]-1)/2  # x center : float
        M = cv.getRotationMatrix2D((cx, cy), self.rotate, 1)  # rotation matrix
    
        # Affine transformation to rotate the image and output size s[1],s[0]
        return cv.warpAffine(img, M, (s[1], s[0]))
    
class Crop(object):
    """Crop image prior to ToTensor step"""
    def __call__(self, sample):
        img = sample["image"]
         # Crop middle part of image1
        shape1 = img.shape[-2:]
        h1 = shape1[0] // 3
        h2 = shape1[0] // 3 * 2
        w1 = shape1[1] // 3
        w2 = shape1[1] // 3 * 2
        cropped_image = img[:,h1:h2,w1:w2] # channels,height,width
        return cropped_image
                
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample["image"]
        # If the image is grayscale, expand its dimensions to have a third axis
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)
    
class Transforms_SimCLR(object):
    def __init__(self, 
                 blur, 
                 brighten, 
                 translate, 
                 zoom, 
                 rotate, 
                 noise_mean, 
                 noise_std, 
                #  cutout_holes, 
                #  cutout_size, 
                data_dir,
                file_list
                ):
        #print(translate)
        
        self.train_transform = transforms.Compose([
            # Stitch image should happen before the fill voids
            StitchAdjacentImagesVer2(data_dir, file_list),
            # FillVoids(), 
            transforms.RandomApply([H_Flip()], p=0.5),
            transforms.RandomApply([V_Flip()], p=0.5),
            transforms.RandomApply([P_Flip()], p=0.5), 
            transforms.RandomApply([Rotate(rotate)], p=0.5),
            transforms.RandomApply([Brighten(brighten)], p=0.5),
            transforms.RandomApply([Translate(translate)], p=0.5),
            transforms.RandomApply([Zoom(zoom)], p=0.5),
            transforms.RandomApply([Blur(blur)], p=0.5),
            transforms.RandomApply([AddNoise(noise_mean, noise_std)], p=0.5),
            # transforms.RandomApply([Cutout(cutout_holes, cutout_size)], p=1), 
        ToTensor()])
        
        

        self.test_transform = transforms.ToTensor()
    
    def __call__(self, sample):
        # Why are we doing this?
        transformed_image1 = self.train_transform(sample)
        transformed_image2 = self.train_transform(sample)
        return transformed_image1, transformed_image2
    

    
    #transforms = transforms.RandomApply(torch.nn.ModuleList([
    #transforms.ColorJitter(),
    # ]), p=0.3)
    
def main():
    ...
    
if __name__ == "__main__":
    main()