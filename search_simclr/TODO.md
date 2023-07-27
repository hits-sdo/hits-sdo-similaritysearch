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
- [x] profit

## 6/21/23
- [X] fix the fillVoids class line 120 (inPaint & inRange) check the image format (check RGB, HSV ordering) and also check the shape of the image, and datatype (Subhamoy did this but we may need to adapt it)
- [ ] clean out the __getitem__ function in dataset.py, so it only has the self.transforms and the two images it returns
- [ ] test cases for the dataset class
- [ ] pytorch lightning datamodule!!!
- [ ] test the pytorch lightning datamodule
- [ ] merge branch to main by making PR
- [ ] create an issue for the model using lightly
- [ ] create a branch using the issue id number and start creating model
- [ ] profit 💰💰💰💰💰💰

## 6/23/23
- [ ] Return filenames and dummy "label" to make lightly compatible
- [ ] Port main code in dataset.py into unittest
- [X] pytorch lightning datamodule!!!
- [x] test the pytorch lightning datamodule❗
- [ ] merge branch to main by making PR ❗
- [ ] create an issue for the model using lightly
- [ ] create a branch using the issue id number and start creating model
- [X] profit 💰💰💰💰💰💰

## 6/28/23
- [x] test the pytorch lightning datamodule❗
- [X] clean up extra print statements that are not in main.
- [X] Return filenames and dummy "label" to make lightly compatible
- [X] Port main code in dataset.py into unittest
- [X] merge branch to main by making PR ❗
- [X] create an issue for the model about lightly
- [ ] introduction to weights and biases
- [X] create a branch using the issue id number and start creating model
- [ ] profit 💰💰💰💰💰💰

## 6/30/23
- [X] Port SimCLR LightningModule code
- [ ] Configure SimCLR Lightly code to use pre-trained model
- [ ] Train using NTXentLoss()
- [ ] Configure with Weights & Biases logging
- [ ] Evaluate results using a subset or validation set of data
- [ ] Plot TSNE 2D representation of latent space
- [X] profit 💰💰💰💰💰💰

## 7/3/23
- [ ] Check what code was lost
- [ ] Reincorporate fillVoids and stitchAdjacentImgs (Subhamoy has code)     TBD
- [ ] Configure SimCLR Lightly code to use pre-trained model
- [ ] Train using NTXentLoss()
- [ ] Configure with Weights & Biases logging
- [ ] Evaluate results using a subset or validation set of data
- [ ] Plot TSNE 2D representation of latent space
- [ ] profit 💰💰💰💰💰💰

## 7/5/23
- [X] Complete `partition_tile_dir_train_val()` to return list of file paths for 'train and val
- [ ] Run dataset.py to verify that changes were successful
- [ ] Adapt unit tests (specifically the Setup method)
- [ ] Adapt source code for datamodule.py `setup()` member function to new dataset requirements
- [ ] Adapt unit tests for datamodule.py 
- [ ] Continue with main function for SimCLR
- [ ] PRO-FIT :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

## 7/7/23
### Reflections from 7/5/23:
- The miniset is no longer in the repository, it has been moved to the shared google folder
- Two design options: `simclr/dataloader/datamodule.py` `prepare_data` 
    - Download from google drive OR 
    - Download directly from team Red repo
- [X] Open pickle file successfuly and plot image
    - The problem could be check file paths again 
    - Check if pickle files are corrupted
- [X] Run dataset.py to verify that changes were successful
- [X] Adapt unit tests (specifically the Setup method)
- [ ] Adapt source code for datamodule.py `setup()` member function to new dataset requirements
- [ ] Adapt unit tests for datamodule.py 
- [ ] Continue with main function for SimCLR
- [ ] Configure github actions with tests to run on push
- [ ] PRO-FIT :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

## 7/10/23

- [X] Adapt and run datamodule unit test `test_data_module.py` (adapt source code)
- [ ] FINISH FILL VOIDS and stitch adjacent image `simclr/dataloader/dataset_aug.py`(get an update on what is happening with this)
    - [ ] Update corresponding test cases `simclr/tests/test_augementations.py`
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
# if time permits...
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results
- [ ] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

## 7/12/23

- [ ] FINISH FILL VOIDS and stitch adjacent image `simclr/dataloader/dataset_aug.py`:
    - [ ] [subhammoys interpolate_superimage()] (https://drive.google.com/drive/folders/1taMZDT_PGk9W3evThwx1xwzeuh4KnXvp)
    - [ ] Adapt `interpolate_superimage()` from `search_utils/image_utils.py` to find nearest neighbor tiles from 
        `train_file_list` variable, account for this using `loc` variable in case `loc` is position (seems like it is dawg) 
    - [ ] Add `FillVoids` and `StitchAdjacentImagesVer2` class calls to `Compose` function in pyTorch transforms
    - [ ] Update corresponding test cases `simclr/tests/test_augementations.py`
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
# if time permits...
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results
- [ ] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

## 7/14/23
- [ ] ~~`search_utils/file_utils.py` Add utility functions to transverse new file tree of jpgs from `aia_171_color_1perMonth.tar.gz`~~ 
    - [ ] ~~`get_file_list()`~~
    - [ ] ~~`get_file_list_from_dir()`~~
    - [ ] ~~`get_file_list_from_dir_recursive()`~~
- [ ] `simclr/dataloader/dataset.py` Adapt `partition_tile_dir_train_val()` to use tot_file_list as a list of paths instead of a list of files (We can keep partition_tile_dir) as is 
- [ ] `search_simclr/simclr/scripts/train_val_split.py` Make tot_file_list a list of file full paths, not just file names
    - [ ] ~~Use `get_file_list_from_dir_recursive()` from `search_utils/file_utils.py`~~
- [ ] Update `get_item` in `readimages` to handle jpg file 
- [ ] Rerun `search_simclr/simclr/scripts/train_val_split.py` to verify that changes were successful
- [ ] Confirm that the Fill Voids and Stitch Adjacent Images are working properly
- [ ] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

# if time permits...
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results

## 7/17/2023
- [X] Finish `search_simclr\simclr\scripts\download_tiles.sh`
    - [ ] Mkdir `data/train_val_simclr`
    - [ ] Add retrieve jpg files with paths from `data/AIA211_193_171_Miniset` to `data/train_val_simclr/tot_file_list.txt`
    - [ ] Fix Line 36 of `scripts/train_val_split.py` to use `data/train_val_simclr/tot_file_list.txt` instead of `os.listdir(tile_dir)`
    - [ ] Run `scripts/train_val_split.py` to create `data/train_val_simclr/train_file_list.txt` and `data/train_val_simclr/val_file_list.txt`
- [ ] `simclr/dataloader/dataset.py` Adapt `partition_tile_dir_train_val()` to use tot_file_list as a list of paths instead of a list of files (We can keep partition_tile_dir) as is 
- [ ] `search_simclr/simclr/scripts/train_val_split.py` Make tot_file_list a list of file full paths, not just file names
- [ ] Update `get_item` in `readimages` to handle jpg file 
- [ ] Rerun `search_simclr/simclr/scripts/train_val_split.py` to verify that changes were successful
- [ ] Confirm that the Fill Voids and Stitch Adjacent Images are working properly
- [x] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

# if time permits...
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results

## 7/19/2023
- [X] Finish `search_simclr\simclr\scripts\download_tiles.sh`
    - [X] Mkdir `data/train_val_simclr`
    - [X] Add retrieve jpg files with paths from `data/AIA211_193_171_Miniset` to `data/train_val_simclr/tot_file_list.txt`
    - [X] Fix Line 36 of `scripts/train_val_split.py` to use `data/train_val_simclr/tot_file_list.txt` instead of `os.listdir(tile_dir)`
    - [ ] Run `scripts/train_val_split.py` to create `data/train_val_simclr/train_file_list.txt` and `data/train_val_simclr/val_file_list.txt`
- [ ] `simclr/dataloader/dataset.py` Adapt `partition_tile_dir_train_val()` to use tot_file_list as a list of paths instead of a list of files (We can keep partition_tile_dir) as is 
- [ ] `search_simclr/simclr/scripts/train_val_split.py` Make tot_file_list a list of file full paths, not just file names
- [ ] Update `get_item` in `readimages` to handle jpg file 
- [ ] Rerun `search_simclr/simclr/scripts/train_val_split.py` to verify that changes were successful
- [ ] Confirm that the Fill Voids and Stitch Adjacent Images are working properly
- [x] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

# if time permits...
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results

## 7/21/2023
- Observed the following errors:
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

- [ ] Run `scripts/train_val_split.py` to create `data/train_val_simclr/train_file_list.txt` and `data/train_val_simclr/val_file_list.txt`
- [ ] `simclr/dataloader/dataset.py` Adapt `partition_tile_dir_train_val()` to use tot_file_list as a list of paths instead of a list of files (We can keep partition_tile_dir) as is 
- [ ] `search_simclr/simclr/scripts/train_val_split.py` Make tot_file_list a list of file full paths, not just file names
- [ ] Update `get_item` in `readimages` to handle jpg file 
- [ ] Rerun `search_simclr/simclr/scripts/train_val_split.py` to verify that changes were successful
- [ ] Confirm that the Fill Voids and Stitch Adjacent Images are working properly
- [x] Profit :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag: :moneybag:

# if time permits...
- [ ] Finish main function for `simclr/model/simCLR.py` with updated datamodule invocation
- [ ] Configure github actions with tests to run on push
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results


## 7/24/2023
- [ ] Continue testing the augmentations in `dataset_aug.py` 
    - [x] `P_Flip()`
    - [x] `Rotate()`
        - [ ] Make the rotations random, and not just 90 clockwhise
    - [x] `Brighten()`
    - [x] `Translate()`
    - [ ] `Zoom()`
    - [ ] `Blur()`
    - [ ] `AddNoise()`
# if time permits...
    - [ ] `Cutout()`
    - [ ] Make the transformations random for `brighten`, `translate`, `zoom`, `blur`, `noise_mean`, `noise_std`
- [ ] Train model
- [ ] Validate model using `val_file_list.txt` created and stored in miniset
- [ ] Create visualizations using clustering in the embedding space to analyze results

## 7/26/2023
-[X] `Zoom()` in `dataset_aug.py` line numbers 238-255 check s (transformed image dimension and original image dimension) against test cases written by team yellow (pasted in line 257)
https://github.com/hits-sdo/hits-sdo-packager/tree/main/sdo_augmentation
notes: in unittests, zoom range is 0 < x < 1 but in the GUI backend
it's 0.5 < x < 5
- [X] `Blur()`
- [X] `AddNoise()`
- [X] `Cutout()`
- [ ] Make the transformations random for `brighten`, `translate`, `zoom`, `blur`, `noise_mean`, `noise_std`
- [ ] Train model
  - [ ] Configure model backbone to utilize pre-trained of the shelf model
  - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
  - [ ] Configure model from scratch
  - [ ] set up wandb
  - [ ] refactor code to log loss and accuracy
- profitttt

## 7/28/2023
- [ ] Create directories at the following path `search_simclr/visualizations/simclr_knn` and `search_simclr/model_weights`
- [ ] configure the `model_run.py` to run the model
- [ ] fix the `Rotate` class in `dataset_aug.py` to make the rotations random
- [ ] Train model
  - [ ] Configure model backbone to utilize pre-trained of the shelf model
  - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
  - [ ] Configure model from scratch
  - [ ] set up wandb
  - [ ] refactor code to log loss and accuracy
- profitttt