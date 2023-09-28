import unittest
import numpy as np
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))
import os
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
        cls.sample = {"image": cls.img, "filename": read_path}
        cls.blur = Blur((3, 3))
        cls.h_flip = H_Flip() 
        cls.v_flip = V_Flip()
        cls.p_flip = P_Flip()
        cls.brighten = Brighten(1.2)
        cls.translate = Translate((20, 20))
        cls.zoom = Zoom(1.5) 
        cls.rotate = Rotate(30.0)
        cls.to_tensor = ToTensor()

    def test_blur(self):
        img_blurred = self.blur({"image": self.img})["image"]
        self.assertLess(img_blurred.std(), self.img.std(), "Image is not blurred")
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_blurred)
        plt.show()
        
    def test_h_flip(self):
        img_hflip = self.h_flip({"image": self.img})["image"]
        assert np.array_equal(img_hflip[:,::-1,:], self.img), "Image is not flipped horizontally"

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_hflip)
        plt.show()

    def test_v_flip(self):
        img_vflip = self.v_flip({"image": self.img})["image"]
        assert np.array_equal(img_vflip[::-1,:,:], self.img), "Image is not flipped vertically"

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_vflip)
        plt.show()

    def test_p_flip(self):
        img_pflip = self.p_flip({"image": self.img})["image"]
        assert np.array_equal(1 - self.img, img_pflip), "Image is not flipped"

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_pflip)
        plt.show()

    def test_brighten(self):
        img_brightened = self.brighten({"image": self.img})["image"]
        assert img_brightened.max() > self.img.max(), "Image is not brightened"

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_brightened)
        plt.show()

    def test_translate(self):
        img_translated = self.translate({"image": self.img})["image"]
        
        shift = np.abs(np.argmax(img_translated, axis=1) - np.argmax(self.img, axis=1))
        assert (shift > 10).any(), "Image is not translated"

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_translated)
        plt.show()

    def test_zoom(self):
        img_zoomed = self.zoom((self.sample))["image"]
        assert img_zoomed.shape[0] > self.img.shape[0], f'Image is not zoomed: {img_zoomed.shape[0]} <= {self.img.shape[0]}'
        assert img_zoomed.shape[1] > self.img.shape[1], f'Image is not zoomed: {img_zoomed.shape[1]} <= {self.img.shape[1]}'
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_zoomed)
        plt.show()

    def test_rotate(self):
        img_rotated = self.rotate({"image": self.img})["image"]
        
        # Check if center portion is same after rotate
        cen_patch = self.img[128:384, 128:384, :] 
        cen_rotated = img_rotated[128:384, 128:384, :]
        
        assert np.allclose(cen_patch, cen_rotated, atol=1e-3), "Center patch is not the same after rotation"
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.img)
        axarr[1].imshow(img_rotated)
        plt.show()


if __name__ == '__main__':
    unittest.main()