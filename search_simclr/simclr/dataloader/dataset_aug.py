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
from typing import Tuple

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
        assert isinstance(value[0], int)
        assert isinstance(value[1], int)
              
        
    def __call__(self, sample):
        """
        Blurs the image by the amount by blur (default = (1, 1))
        Blurring is performed as an average blurring
        with kernel size defined by blur
    """
        image, fname = sample["image"], sample["filename"]
        kernal_w = random.randint(1, self.blur[0])
        kernal_h = random.randint(1, self.blur[1])
        blur_image = cv.blur(image, (kernal_w, kernal_h), 0)
        return {"image": blur_image, "filename": fname}

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
        img, fname = sample["image"], sample["filename"]
        self.std = random.uniform(0, self.std_lim)
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 1)
        return {"image": img, "filename": fname}
    
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

    def __init__(self, data_dir, file_list) -> None:
        self.data_dir = data_dir
        #self.file_name = file_name
        self.file_list = file_list

    def __call__(self, sample):
        
        """
        stitches adjacent images to return a superimage
        """
        image, fname = sample["image"], sample["filename"]
        superImage = stitch_adj_imgs(self.data_dir, 
                        #self.file_name,
                        fname,
                        EXISTING_FILES=self.file_list,
                        multi_wl=True,
                        iterative=True,
                        remove_coords=False)
        # print(f'superImage shape: {superImage.shape}')
       

        return {"image": superImage, "filename": fname}

class Cutout(object):
    def __init__(self, cutout_holes, cutout_size):
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size
        assert isinstance(cutout_holes, int)
        assert isinstance(cutout_size, float)
        

    def __call__(self, sample):
        cutout_image, fname = sample["image"], sample["filename"]
        super_image_height, super_image_width = (cutout_image.shape[:2])
        
        image_height, image_width = (super_image_height // 3, super_image_width // 3)
        
        nholes = random.randint(1, self.cutout_holes)
        
        for _ in range(nholes):
            hole_height = random.randint(int(image_height * self.cutout_size/10.0), int(image_height * self.cutout_size))
            hole_width = random.randint(int(image_height * self.cutout_size/10.0), int(image_width * self.cutout_size))
        
            random_coordinate = (random.randint(image_height, 2*image_height-hole_height), random.randint(image_width, 2*image_width-hole_width))
        
            mask = np.ones_like(cutout_image)

            # Set the region of the hole in the mask to 0
            mask[random_coordinate[0]:random_coordinate[0]+hole_height, random_coordinate[1]:random_coordinate[1]+hole_width] = 0

            # Apply the mask to the image
            cutout_image = cutout_image * mask

        return {"image": cutout_image, "filename": fname}

class H_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        return {"image": cv.flip(image, 1), "filename": fname}
        
class V_Flip(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        return {"image": cv.flip(image, 0), "filename": fname}
    
class P_Flip(object):
    def __init__(self):
        ...
        
    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        return {"image": (1-image), "filename": fname}
    
class Brighten(object):
    def __init__(self, value) -> None:
        self.brighten = value
        assert isinstance(value, float)

    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        min_value = random.uniform(0.3, self.brighten)
        return {"image": np.abs(image)**self.brighten, "filename": fname}
    
class Translate(object):
    def __init__(self, value):
        self.translate = value
        assert isinstance(value[0], int)
        assert isinstance(value[1], int) 
    
    def __call__(self, sample):
        random_x = random.randint(-self.translate[0], self.translate[0])
        random_y = random.randint(-self.translate[1], self.translate[1])
        image, fname = sample["image"], sample["filename"]
        s = image.shape
        m = np.float32([[1, 0, random_x], [0, 1, random_y]])
        # Affine transformation to translate the image and output size
        image = cv.warpAffine(image, m, (s[1], s[0]))
        return {"image": image, "filename": fname}

class Zoom(object):
    def __init__(self, value):
        self.zoom = value
        assert isinstance(value, float)
    
    def __call__(self, sample):
        random_zoom = self.zoom #1.5 #random.uniform(0, self.zoom)
        image, fname = sample["image"], sample["filename"] #Unpack the dictionary
        original_image_shape = image.shape
        zoomed_immage_shape = (int(random_zoom*original_image_shape[0]), int(random_zoom*original_image_shape[1]))
        img_zeros = np.zeros(original_image_shape) #temporary empty canvas the sizer of the original image

        image_resize = cv.resize(image, (zoomed_immage_shape[1], zoomed_immage_shape[0]), interpolation=cv.INTER_CUBIC)
        # Resize the image using zoom as scaling factor with area interpolation
        if random_zoom < 1:
            y1 = original_image_shape[0]//2 - zoomed_immage_shape[0]//2 #center of originall image - half of zoomed image
            y2 = original_image_shape[0]//2 + zoomed_immage_shape[0]//2 #center of originall image + half of zoomed image
            x1 = original_image_shape[1]//2 - zoomed_immage_shape[1]//2
            x2 = original_image_shape[1]//2 + zoomed_immage_shape[1]//2
            img_zeros[y1:y2, x1:x2] = image_resize #inlay the "ZOOMED OUT" - Actually just shrunk immage inside the zeros 
        else:
            y1 = zoomed_immage_shape[0]//2 - original_image_shape[0]//2 #Center of zoomed image - half of original image
            y2 = zoomed_immage_shape[0]//2 + original_image_shape[0]//2 #Center of zoomed image + half of original image
            x1 = zoomed_immage_shape[1]//2 - original_image_shape[1]//2
            x2 = zoomed_immage_shape[1]//2 + original_image_shape[1]//2
            img_zeros = image_resize[x1:x2, y1:y2,:] #the zeroes immage now gets the cutout of the "ZOOMED IN" immage - Actually just expanded immage
            
        return {"image": img_zeros, "filename": fname} #Repac and return the dictionary

'''    def test_zoom(self):
        """visually check zooming in and out"""
        z = 2
        image_tr = self.augmentations.zoom(self.image, zoom=z)
        s_tr = image_tr.shape
        s_old = self.image.shape
        y1 = s_tr[0]//2 - s_old[0]//2
        y2 = s_tr[0]//2 + s_old[0]//2
        x1 = s_tr[1]//2 - s_old[1]//2
        x2 = s_tr[1]//2 + s_old[1]//2
        image_tr = image_tr[y1:y2, x1:x2]
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(image_tr, vmin=0, vmax=1)
        if z < 1:
            plt.title('zoomed out image')
        else:
            plt.title('zoomed in image')
        plt.show()'''
        
class Rotate(object):
    def __init__(self, value=360.0):
        self.rotate = value #this is a tuple
        # print(f"rotate: {self.rotate}") 
        assert isinstance(value, float)

    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        s = image.shape
        cy = (s[0]-1)/2  # y center : float
        cx = (s[1]-1)/2  # x center : float
        M = cv.getRotationMatrix2D((cx, cy), random.uniform(0.0, self.rotate), 1)  # rotation matrix
    
        # Affine transformation to rotate the image and output size s[1],s[0]
        return {"image": cv.warpAffine(image, M, (s[1], s[0])), "filename": fname}
    
class Crop(object):
    """Crop image prior to ToTensor step"""
    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
         # Crop middle part of image1
        shape1 = image.shape[:2]
        h1 = shape1[0] // 3
        h2 = shape1[0] // 3 * 2
        w1 = shape1[1] // 3
        w2 = shape1[1] // 3 * 2
        cropped_image = image[w1:w2,h1:h2,:] #width, height, channels
        return {"image": cropped_image, "filename": fname}

class ReSize(object):
    def __init__(self, resize_height: int, resize_width: int ) -> None:
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        resized_image = cv.resize(image, (self.resize_width, self.resize_height))
        return {"image": resized_image, "filename": fname}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, fname = sample["image"], sample["filename"]
        # If the image is grayscale, expand its dimensions to have a third axis
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image).to(torch.float32), "filename": fname}
    
class Transforms_SimCLR(object):
    def __init__(self, 
                 blur, 
                 brighten, 
                 translate, 
                 zoom, 
                 rotate, 
                 noise_mean, 
                 noise_std, 
                cutout_holes, 
                cutout_size, 
                data_dir,
                file_list
                ):
        #print(translate)
        
        self.train_transform = transforms.Compose([
            # Stitch image should happen before the fill voids
            StitchAdjacentImagesVer2(data_dir, file_list),
            transforms.RandomApply([H_Flip()], p=0.5),
            transforms.RandomApply([V_Flip()], p=0.5),
            transforms.RandomApply([P_Flip()], p=0.5), 
            transforms.RandomApply([Rotate(rotate)], p=0.5),
            transforms.RandomApply([Brighten(brighten)], p=0.5),
            transforms.RandomApply([Translate(translate)], p=0.5),
            transforms.RandomApply([Zoom(zoom)], p=0.5),
            transforms.RandomApply([Cutout(cutout_holes, cutout_size)], p=0.5),
            transforms.RandomApply([Blur(blur)], p=0.5),
            transforms.RandomApply([AddNoise(noise_mean, noise_std)], p=0.5),
            Crop(),
        ToTensor()])
        
        

        self.test_transform = transforms.ToTensor()
    
    def __call__(self, sample):
        transformed_sample1 = self.train_transform(sample)
        transformed_sample2 = self.train_transform(sample)
        return transformed_sample1, transformed_sample2
    

    
    #transforms = transforms.RandomApply(torch.nn.ModuleList([
    #transforms.ColorJitter(),
    # ]), p=0.3)
    
def main():
    ...
    
if __name__ == "__main__":
    main()