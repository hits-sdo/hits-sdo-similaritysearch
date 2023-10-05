import unittest
import numpy as np
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import os
import cv2
# import simclr.dataloader.dataset
# from search_simclr.simclr.dataloader import dataset_aug
from search_utils.image_utils import (
    read_image
)
from search_simclr.simclr.dataloader.dataset_aug import (
    Blur,
    H_Flip,
    V_Flip,
    P_Flip,
    Brighten,
    Translate,
    Zoom,
    Rotate,
    ToTensor
)
import matplotlib.pyplot as plt

class TestAugmentations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.img = np.random.rand(256, 256, 3).astype(np.float32)
        #C:\Users\jvigi\hitssdo\hits-sdo-similaritysearch\data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles\20100601_000008_aia_211_193_171_tile_256_512.jpg
        read_path = os.path.join(root, "data", "nadia.png")
        cls.img = read_image(read_path, 'png')
        # print(f'cls.img = {cls.img}')
        cls.sample = {"image": cls.img, "filename": read_path}
        cls.blur = Blur((10, 10))
        cls.h_flip = H_Flip() 
        cls.v_flip = V_Flip()
        cls.p_flip = P_Flip()
        cls.brighten = Brighten(0.2)
        cls.translate = Translate((20, 20))
        cls.zoom = Zoom(1.5) 
        cls.rotate = Rotate(30.0)
        cls.to_tensor = ToTensor()
        cls.plotImages = False

    def test_blur(self):
        sample_img_blurred = self.blur(self.sample)
        img_blurred = sample_img_blurred["image"]
        print(type(img_blurred))
        self.assertLess(img_blurred.std(), self.img.std(), "Image is not blurred")
        
        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_blurred)
            axarr[1].set_title('Blurred Image')
            plt.show()
        
    def test_h_flip(self):
        sample_img_hflip = self.h_flip(self.sample)
        img_hflip = sample_img_hflip["image"]
        assert np.array_equal(img_hflip[:,::-1,:], self.img), "Image is not flipped horizontally"

        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_hflip)
            axarr[1].set_title('H-Flipped Image')
            plt.show()

    def test_v_flip(self):
        sample_img_vflip = self.v_flip(self.sample)
        img_vflip = sample_img_vflip["image"]
        assert np.array_equal(img_vflip[::-1,:,:], self.img), "Image is not flipped vertically"

        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_vflip)
            axarr[1].set_title('V-Flipped Image')
            plt.show()

    def test_p_flip(self):
        sample_img_pflip = self.p_flip(self.sample)
        img_pflip = sample_img_pflip["image"]
        assert np.array_equal(1 - self.img, img_pflip), "Image is not flipped"

        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_pflip)
            axarr[1].set_title('P-Flipped Image')
            plt.show()

    def test_brighten(self):
        sample_img_brightened = self.brighten(self.sample)
        img_brightened = sample_img_brightened["image"]
        
        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_brightened)
            axarr[1].set_title('Brightened Image')
            plt.show()       

        print(f"img_brightened.min() = {np.average(img_brightened)}, self.img.min() = {np.average(self.img)}") 

        assert np.average(img_brightened) != np.average(self.img), "Image is not brightened"



    def test_translate(self):
        sample_img_translated = self.translate(self.sample)
        img_translated = sample_img_translated["image"]
        
        shift = np.abs(np.argmax(img_translated, axis=1) - np.argmax(self.img, axis=1))
        assert (shift > 10).any(), "Image is not translated"

        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_translated)
            axarr[1].set_title('Translated Image')
            plt.show()

    def test_zoom(self):
        sample_img_zoomed = self.zoom(self.sample)
        img_zoomed = sample_img_zoomed["image"]
        
        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_zoomed)
            axarr[1].set_title('Zoomed Image')
            plt.show()
        
        assert img_zoomed.shape[0] == self.img.shape[0], f'Zoomed image is not the same dimensions as original: {img_zoomed.shape[0]} != {self.img.shape[0]}'
        assert img_zoomed.shape[1] == self.img.shape[1], f'Zoomed image is not the same dimensions as original: {img_zoomed.shape[1]} != {self.img.shape[1]}'
        


    def test_rotate(self):
        sample_img_rotated = self.rotate(self.sample)
        img_rotated = sample_img_rotated["image"]
        
        # Check if center portion is same after rotate
        height, width, _ = self.img.shape
        center_y = height // 2
        center_x = width // 2
        cen_patch = self.img[center_y-64:center_y+64, center_x-64:center_x+64, :]
        cen_rotated = img_rotated[center_y-64:center_y+64, center_x-64:center_x+64, :]

        # Split the image into its color channels
        # the format for images in opencv is 
        np_image = cen_patch.astype(np.uint8)
        bgr_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        blue_hist = cv2.calcHist([bgr_img], [0], None, [256], [0, 256])
        green_hist = cv2.calcHist([bgr_img], [1], None, [256], [0, 256])
        red_hist = cv2.calcHist([bgr_img], [2], None, [256], [0, 256])

        np_rot_image = cen_rotated.astype(np.uint8)
        bgr_rot_img = cv2.cvtColor(np_rot_image, cv2.COLOR_RGB2BGR)
        blue_rot_hist = cv2.calcHist([bgr_rot_img], [0], None, [256], [0, 256])
        green_rot_hist = cv2.calcHist([bgr_rot_img], [1], None, [256], [0, 256])
        red_rot_hist = cv2.calcHist([bgr_rot_img], [2], None, [256], [0, 256])
        
        if self.plotImages:
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(self.img)
            axarr[1].imshow(img_rotated)
            axarr[1].set_title('Rotated Image')
            plt.show()

        assert((cv2.compareHist(blue_hist, blue_rot_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.9) and \
            (cv2.compareHist(green_hist, green_rot_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.9) and \
            (cv2.compareHist(red_hist, red_rot_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.9)), 'Center patch is not the same after rotation'

        
        
        '''
        The np.allclose function returns True if two arrays are element-wise equal within a tolerance. 
        The tolerance level is defined by the atol parameter, which in this case is 1e-3. If the arrays 
        are not close, np.allclose returns False.
        '''
        # assert np.allclose(cen_patch, cen_rotated, atol=1e-3), "Center patch is not the same after rotation"
        



if __name__ == '__main__':
    unittest.main()