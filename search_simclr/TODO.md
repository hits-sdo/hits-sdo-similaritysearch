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