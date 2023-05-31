import unittest
import numpy as np
import pyprojroot
root = pyprojroot.here()
import simclr.dataloader.dataset
from simclr.dataloader.dataset_aug import (
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
import sys
sys.path.append(str(root))
import matplotlib.pyplot as plt

class test_augmentations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.rand(1024, 1024).astype(np.uint8)
        cls.blur = Blur((2,2))
        cls.h_flip = H_Flip((True))
        cls.v_flip = V_Flip((True))
        cls.p_flip = P_Flip((True))
        cls.brighten = Brighten((1.5))
        cls.translate = Translate((10, 10))
        cls.zoom = Zoom(1.2)
        cls.rotate = Rotate(1.8)
        cls.to_tensor = ToTensor()


    @classmethod
    def test_brighten(cls):
        img_brightened = cls.brighten(cls.img)
        # assert img_brightened.max() > cls.img.max()
        plt.subplot(1, 2, 1)
        plt.imshow(cls.img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_brightened)
        
        plt.show()
        pass
    
    '''  superImage = stitch_adj_imgs(self.DATA_DIR, self.FILE_NAME)
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, vmin=0, vmax=1)
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(superImage, vmin=0, vmax=1)
        plt.title('superimage')
        plt.show()



        if __name__ == '__main__':
        unittest.main()
        '''
    @classmethod
    def test_translate(cls):
        
        pass
    
    @classmethod
    def test_zoom(cls):
        pass
    
    @classmethod
    def test_rotate(cls):
        pass
    
    @classmethod
    def test_h_flip(cls):
        
        pass
    
    @classmethod
    def test_v_flip(cls):
        pass
    
    @classmethod
    def test_blur(cls):
        pass
    
    @classmethod
    def test_p_flip(cls):
        pass


if __name__ == '__main__':
    unittest.main()