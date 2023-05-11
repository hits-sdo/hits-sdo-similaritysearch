from sdo_augmentation.augmentation import Augmentations
import sys
sys.path.append('search-utils')
from image_utils import read_image, stitch_adj_imgs

import glob
import json
import matplotlib.pyplot as plt

file_dir = "./data/miniset/AIA171/monochrome/"

file_list = glob.glob(file_dir + "*.p")

img = read_image(file_list[60], 'p')
data_js = json.load(open('./data/Augmentationfile_EUV.json'))

print(data_js.keys())

a1 = Augmentations(img, data_js["Augmentation_1"][0])
a2 = Augmentations(img, data_js["Augmentation_2"][0])

aug_img1, _ = a1.perform_augmentations()
aug_img2, _ = a2.perform_augmentations()

size_tuple_img = (1, 3)

plt.subplot(*size_tuple_img, 1)
plt.imshow(img)

plt.subplot(*size_tuple_img, 2)
plt.imshow(aug_img1)

plt.subplot(*size_tuple_img, 3)
plt.imshow(aug_img2)
plt.show()


file_names = [x[len(file_dir):] for x in file_list]

print(file_names)
