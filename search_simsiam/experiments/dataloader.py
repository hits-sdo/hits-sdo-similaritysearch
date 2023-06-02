from sdo_augmentation.augmentation import Augmentations
import sys
sys.path.append('./')
print(sys.path)
from search_utils.image_utils import read_image, stitch_adj_imgs

import glob
import json
import matplotlib.pyplot as plt


CURRENT_IMG_IDX = 30


def main():
    file_dir = "./data/miniset/AIA171/monochrome/"

    file_list = glob.glob(file_dir + "*.p")

    img = read_image(file_list[CURRENT_IMG_IDX], 'p')
    data_js = json.load(open('./data/Augmentationfile_EUV.json'))

    print(data_js.keys())

    a1 = Augmentations(img, data_js["Augmentation_1"][0])
    a2 = Augmentations(img, data_js["Augmentation_2"][0])

    aug_img1, _ = a1.perform_augmentations()
    aug_img2, _ = a2.perform_augmentations()

    size_tuple_img = (2, 3)
    figsize = (2*5, 3*5)  # 15x10

    figure, axis = plt.subplots(*size_tuple_img, figsize=figsize)
    axis = axis.ravel()

    images = [img, aug_img1, aug_img2]

    file_names = [x[len(file_dir):] for x in file_list]

    super_image = stitch_adj_imgs(data_dir=file_dir,
                                  file_name=file_names[CURRENT_IMG_IDX],
                                  EXISTING_FILES=file_names)

    as1 = Augmentations(super_image, data_js["Augmentation_1"][0])
    as2 = Augmentations(super_image, data_js["Augmentation_2"][0])

    sup_aug_img1, _ = as1.perform_augmentations()
    sup_aug_img2, _ = as2.perform_augmentations()

    # Get size of img
    img_h, img_w = img.shape[:2]

    # Get size of super image
    sup_img_h, sup_img_w = super_image.shape[:2]

    # grab center of parent/super image
    center_sup_img_y, center_sup_img_x = (sup_img_h // 2, sup_img_w // 2)

    # grab center of tile
    center_img_y, center_img_x = (img_h // 2, img_w // 2)

    sup_aug_img1 = sup_aug_img1[
        center_sup_img_y - center_img_y:
        center_sup_img_y + center_img_y,
        center_sup_img_x - center_img_x:
        center_sup_img_x + center_img_x
        ]

    sup_aug_img2 = sup_aug_img2[
        center_sup_img_y - center_img_y:
        center_sup_img_y + center_img_y,
        center_sup_img_x - center_img_x:
        center_sup_img_x + center_img_x
        ]

    for image in [img, sup_aug_img1, sup_aug_img2]:
        images.append(image)

    title_list = ['OG', 'aug_1', 'aug_2', 'OG', 'aug_1_ filled', 'aug_2_filled']
    for image, axis_, title in zip(images, axis, title_list):
        axis_.imshow(image, vmin=0, vmax=1, cmap='gray')
        axis_.set_title("image_" + title)

    plt.show()



if __name__ == '__main__':
    main()
