import dataset
import unittest
import dataset_aug
import numpy as np
import pyprojroot
root = pyprojroot.here()
import sys
sys.path.append(str(root))

class test_augmentations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = np.random.rand(1024, 1024).astype(np.uint8)
        cls.Blur((2,2))
        cls.H_Flip((True))
        cls.V_Flip((True))
        cls.P_Flip((True))
        cls.Brighten((1.5))
        cls.Translate((10, 10))
        cls.Zoom(1.2)
        cls.Rotate(1.8)
        cls.ToTensor()


    @classmethod
    def test_brighten(cls):
        pass
    
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
    