# 5/12/23
## Todo List:
- Designing Dataset Class
    - Constructor
        - .csv containing file names (pickle file names)(or txt)
            - also json file names(or another .csv file)
        - contains all dataset attributes (label name,
                                            label dir, img dir, img name, img size, image file, img transform {object})
    - __getitem__()
        - store np array 
        - use the array with the filenames from .csv to load the pickle file and assign its output to np array
        - transform to tensor
    - __len__()
        - return the length of the np array


## 5/12/23
## Consideration For Next Meeting
- Random augmentations on a tile saved in json files after augmentation (parallel arrays)
    -Naming convention - same name, different type augmentation (buddy naming, json + tile thats already augmented)
- OR
- Random augmentations in the data loader, by taking advatage of the composed transform function in pytorch
    - If so, refactor random augs from team yellow, or load the json file that were generated

## 5/17/23 

## closing thoughts
    
    - Implement augmentation functions from team yellow in order to test labels and ability to call augmented images from file path 
        - Problem: how to import?
        - Where do put? 
    - Decrease mob time? for more rotations
    - Retro in between: do we wanna keep going with it? (might be a good little break)
    - Keep in mind: loading random augs or create them on the fly? (getitem)
        - return tuple (image, dictionary)

## 5/24/23 
- [x] Returned an image from the getitem function in dataset
- [x] Fixed filepaths
- [x] Imported team yellow augmentation classes
- [x] Got an appropriate list of files from the miniset folder
- [x] Started defining callable classes for Transform

TODO: 
- [X] Complete the transform classes
- [ ] Finish some more test cases
    - Test cases for dataset class
    - Test cases for the transforms
- [ ] Implement the pytorch Lightning data module (tutorial pls)
- [X] Profit >:D

## 5/31/23
- [ ] Finish the Test Cases for Augmentations
- [ ] Finish the __getitem__ with RandomTransforms
- [ ] Test using dataloader
- [ ] Port dataloader boiler plate code to PyTorch Lightning
- [X] CONDA ENVIRONMENT
- [X] Profit ::moneybag::

## 6/2/23
- [ ] FINISH THE TEST CASES FOR AUGMENTATIONS 
 - Numpy.allclose (one line to check 2D arrays for equality)
- [ ] __getitem__ w/ random transforms 
- [ ] Port dataloader boiler plate code to PyTorch Lightning
- [ ] Test using dataloader
- [ ] EVERYONE INSTALL CONDA ENVIROMENT
- [ ] PROFIT ( big money dollaz ) ::moneybag::

## 6/5/23
- [ ] Add augmentation to the dataset_aug class
    - [ ] Add random noise augmentation to the dataset_aug class
    - [ ] Add random cutout augmentation to the dataset_aug class
- [X] Fix dataset import

## 6/7/23
- [X] Make dataset.py runnable by fixing the errors
- [ ] Add augmentation to the dataset_aug class
    - [ ] Add random noise augmentation to the dataset_aug class
    - [ ] Add random cutout augmentation to the dataset_aug class
- [ ] Port dataloader boiler plate code to PyTorch Lightning
- [XX] Profit!!!!

## 6/9/23
- [X] Add augmentation to the dataset_aug class
    - [X] Add random noise augmentation to the dataset_aug class
    - [X] Add random cutout augmentation to the dataset_aug class
- [ ] Port dataloader boiler plate code to PyTorch Lightning
- [ ] Return 2 images + filename (potentially? - Lightly pkg thing)
- [ ] Pull request - new branch for modeling!
- [X] Profit!!!💰💰💰💰

## 6/14/23
- [ ] Vote which design to use (Pytorch, or Lightning)
- [ ] Test Cutout Augmentation
- [ ] Port everything to Pytorch Lightning Data Module
- [ ] If everything works, PR to main
    - [ ] Open an issue to dev the model
    - [ ] Create a new branch "SUNFLOWER_SIMCLR_MODEL" with the issue ID
- [ ] Simclr Model (Adapt lightly model to our dataset class)
- [ ] Wandb (Weights and Bias)
- [ ] Profit!!!💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰💰

## 6/16/23
- [x] added FillVoids class for use with compose function from pytorch transforms
- [x] added StitchAdjacentImagesVer2 class for use with compose function from pytorch transforms
- [ ] test fillVoids and StitchAdjacentImagesVer2 using the transforms_Simclr compose function
- [ ] add pytorch data module 
- [ ] test pytorch data module
- [ ] merge branch to main by making PR
- [ ] create an issue for the model using lightly
- [ ] create a branch using the issue id number and start creating model
- [ ] profit