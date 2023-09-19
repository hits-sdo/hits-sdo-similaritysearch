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
- [X] Create directories at the following path `search_simclr/visualizations/simclr_knn` and `search_simclr/model_weights`
- [ ] configure the `model_run.py` to run the model
    - [X] complete `wandb.init()` function call on line 52
- [X] fix the `Rotate` class in `dataset_aug.py` to make the rotations random
- [ ] Train model
  - [ ] Configure model backbone to utilize pre-trained of the shelf model
  - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
  - [ ] Configure model from scratch
  - [ ] set up wandb
  - [ ] refactor code to log loss and accuracy
- profitttt

## 7/31/23
- [ ] configure the `model_run.py` to run the model
    - [ ] wandb.login()
- [X] Update in 'model_run.py' in 'SDOConfig' to include file lists
    - [X] tile_dir
    - [X] train_val_dir
    - [X] train_file_list
    - [X] val_file_list
- [X] See 'train_val_split.py' for more information
- [X] Update DataModule default to include transformations
    - [X] Work in cutout
    - [X] Copy line 50 in 'train_val_split.py' 
- [ ] Train model
    - [ ] Add progress bars 
  - [ ] Configure model from scratch
        - [ ] Configure model backbone to utilize pre-trained of the shelf model
        - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
  - [ ] set up wandb
  - [ ] refactor code to log loss and accuracy
  - [ ] biiiig profit :moneybag: :moneybag: :moneybag:

## 8/2/23
- [ ] configure the `model_run.py` to run the model
    - [x] ~~wandb.login()~~ Login use wandb.login() in the terminal and it'll prompt you to login
    - [x] Test wandb with the example [quickstart](https://docs.wandb.ai/quickstart)
- [ ] Train model
    - [ ] Add progress bars 
- [ ] Configure model from scratch
    - [ ] Configure model backbone to utilize pre-trained of the shelf model
    - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
- [x] set up wandb
- [ ] refactor code to log loss and accuracy
- [X] biiiig profit :moneybag: :moneybag: :moneybag:

## 8/7/23
- [X] In `model_run.py` train_flist is not the list of file names, but instead the file name of the .txt file. We need to call `get_file_list()` to get the list of file names
- [X] Rename `train_flist` to `train_fpath` and `val_flist` to `val_fpath` and `test_flist` to `test_fpath`
- [X] Create `train_flist = get_file_list(train_flist_name)` and `val_flist = get_file_list(val_flist_name)` and `test_flist = get_file_list(test_flist_name)`
- [ ] Rerun `model_run.py` to verify file paths with a breakpoint
- [ ] Remove breakpoint
- [ ] Configure validation step to run on the validation set
    - [ ] Uncomment `generate_embeddings()` in `model_run.py` line 130
    - [ ] Uncomment `plot_knn_examples()` in `model_run.py` line 131
        - [ ] Save plots to file in `plot_knn_examples()` instead of showing to the screen
- [ ] Test `model_run.py` to run the model with 50 examples
- [ ] Train model
    - [ ] Add progress bars 
- [ ] Configure model from scratch
    - [ ] Configure model backbone to utilize pre-trained of the shelf model
    - [ ] [Tutorial reference](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
- [ ] refactor code to log loss and accuracy
- [x] biiiig profit :moneybag: :moneybag: :moneybag:

## 8/9/23
- [x] Add code from `train_val_split.py` line 30-46, and move them to `model_run.py`, just before get_file_list
- [X] Add hyperparameters to `model_run.py` from [Tutorial Reference](https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/train.py)
- [x] in `model_run.py` `ValueError: wandb.log must be passed a dictionary`
- [x] save the model training in `model_run.py`
- [ ] expand the number of files in `train_val_split.py` to be larger then 50
- [ ] Hyper-param sweep `.yml` file for different learning rates, sampling rates and batch sizes
    [Tutorial Reference](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] biiiig profit :moneybag: :moneybag: :moneybag:

## 8/14/23
- [X] comment out old config and replace with new config variables
- [ ] cutout some holes
- [ ] Hyper-param sweep `.yml` file for different learning rates, sampling rates and batch sizes
    [Tutorial Reference](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)
- [ ] train with the hyperparameter sweep
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [x] biiiig profit :moneybag: :moneybag: :moneybag:

## 8/16/23
- [x] complete 'prepare_data()' in 'datamodule.py' to code in 'train_val_split.py'
    - [x] Problem files, from 'split_val_files()' in 'file_utils.py'
    - [ ] Add print statement to check the length of 'train_file_list' and 'val_file_list'
    - [x] Include train_val_split
- [*] Hyper-param sweep `.yml` file for different learning rates, sampling rates and batch sizes
    [Tutorial Reference](https://docs.wandb.ai/guides/sweeps/add-w-and-b-to-your-code)
    [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [ ] train with the hyperparameter sweep
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] cutout some holes
- [x] biiiig profit :moneybag: :moneybag: :moneybag: 💰💰💰💰💰💰💰💰

## 8/21/23
- [X] Fix the bug with the generate_embeddings()
- [X] Add print statement to check the length of 'train_file_list' and 'val_file_list'
- [ ] Watch [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [ ] train with the hyperparameter sweep
- [ ] Save images
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] cutout some holes
- [X] all the profit💰💰💰💰💰💰💰💰

## 8/23/23
- [ ] Watch [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [x] Train with the hyperparameter sweep
- [x] Save images
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] cutout some holes
- [x] write code, get cash 💰💰💰💰💰💰💰💰

## 8/28/23
- [x] Did Code Review for branch `wip-sunflower-sweep-changes`
- [ ] Reference SimSiam: [SimSiam](https://github.com/hits-sdo/hits-sdo-similaritysearch/blob/ss_training/search_simsiam/simsiam_HITS_cleaned_up.ipynb)
    - [ ] `ss_training``
    - [x] Fix ``IndexError: list index out of range`` in ``model_run.py`` line 221
- [ ] Generate embeddings script 'generate_embeddings.py'
    - [ ] Refactor code in ``vis_utils.py`` to be compatible with Simclr
        - [ ] plot_nearest_neighbors_3x3()
        - [ ] get_image_as_np_array_with_frame
        - [ ] get_scatter_plot_with_thumbnails
- [ ] Watch [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [x] cutout some holes
- [ ] randomize augmentations
    - [x] Rotate
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [x] Cutout
    - [ ] Blur
    - [x] Noize
- [x] Create a script `sweeps_run.py`
- [ ] Log more than just the loss to wandb
- [x] SUNFLOWER PROFIT 💰💰💰💰💰💰💰💰

## 8/30/23
- [x] Added different feature extractor variable to the SimCLR model
- [ ] Write Checkpoint code to intercept SimCLR architecture for training images at the output of the feature extractor (resnet101, resnet50, ...check current architecture for feature extractor type)
  - [ ] May want to add feature extractor type to argparser in case user wants to change up the feature extractor
  - [ ] Run WandB sweeps w/ different feature extractor and pick the one with the lowest loss
- [ ] Take output of feature extractor checkpoint and funnel into 2D dimensionality reduction (dot plot):
  - [ ] PCA
  - [ ] TSNE
  - [ ] UMAP
- [ ] Write the functions to plot above
- [ ] Write script to plot the training set embeddings and do the same 
- [X] Fix issue with model_run.py starting offline if ran without using the sweeps
- [ ] ~~Parallel Wandb Agents~~
- [ ] Reference SimSiam: [SimSiam](https://github.com/hits-sdo/hits-sdo-similaritysearch/blob/ss_training/search_simsiam/simsiam_HITS_cleaned_up.ipynb)
    - [ ] `ss_training``
- [ ] Generate embeddings script 'generate_embeddings.py'
    - [ ] Refactor code in ``vis_utils.py`` to be compatible with Simclr
        - [ ] plot_nearest_neighbors_3x3()
        - [ ] get_image_as_np_array_with_frame
        - [ ] get_scatter_plot_with_thumbnails
- [ ] Watch [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] randomize augmentations
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [ ] Log more than just the loss to wandb
- [x] SUNFLOWER PROFIT 💰💰💰💰💰💰💰💰

## 9/1/23
- [x] Write Checkpoint code to intercept SimCLR architecture for training images at the output of the feature extractor (resnet101, resnet50, ...check current architecture for feature extractor type)
  - [ ] Load from checkpoint [Checkpoint](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
  - [ ] May want to add feature extractor type to argparser in case user wants to change up the feature extractor [Hooks and Callbacks](https://stephencowchau.medium.com/pytorch-lightning-hooks-and-callbacks-my-limited-understanding-d8e0a56dcf2b)
  - [ ] Run WandB sweeps w/ different feature extractor and pick the one with the lowest loss
- [ ] Take output of feature extractor checkpoint and funnel into 2D dimensionality reduction (dot plot):
  - [ ] PCA
  - [ ] TSNE
  - [ ] UMAP
- [ ] Write the functions to plot above
- [ ] Write script to plot the training set embeddings and do the same 
- [ ] Generate embeddings script 'generate_embeddings.py'
    - [ ] Refactor code in ``vis_utils.py`` to be compatible with Simclr
        - [ ] plot_nearest_neighbors_3x3()
        - [ ] get_image_as_np_array_with_frame
        - [ ] get_scatter_plot_with_thumbnails
- [ ] Reference SimSiam: [SimSiam](https://github.com/hits-sdo/hits-sdo-similaritysearch/blob/ss_training/search_simsiam/simsiam_HITS_cleaned_up.ipynb)
* If we have time! *
- [ ] Watch [Video Reference](https://www.youtube.com/watch?v=9zrmUIlScdY&ab_channel=Weights%26Biases)
- [ ] Create 'tsne' and 'pca' plots locally using 'matplotlib'
- [ ] Configure WandB to create `tsne` and `pca` plots https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html
- [ ] randomize augmentations
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [ ] Log more than just the loss to wandb 
    - [ ] `collapse`
    - [x] referense SimSiam (https://github.com/hits-sdo/hits-sdo-similaritysearch/blob/main/search_simsiam/simsiam_example_notebook_HITS_reproducible.ipynb)
    - [ ] epochs
    - [ ] accuracy
- [ ] [sweeps in wandb](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
- [ ] [wandb sweeps documentation](https://docs.wandb.ai/guides/sweeps)
- [ ] [wandb w/ simclr](https://wandb.ai/sayakpaul/simclr/reports/Towards-self-supervised-image-understanding-with-SimCLR--VmlldzoxMDI5NDM?_gl=1%2A1leb11n%2A_ga%2AMTAwNzI4MDkxLjE2OTE2MTkxODM.%2A_ga_JH1SJHJQXJ%2AMTY5MzYxMDE0OC4zOS4xLjE2OTM2MTA1NTguNDguMC4w)
- [x] SUNFLOWER PROFIT 💰💰💰💰💰💰💰💰
- [ ] Look at [this:] (https://github.com/hits-sdo/hits-sdo-similaritysearch/blob/main/search_simsiam/simsiam_example_notebook_HITS_reproducible.ipynb)

## 09/08/2023
- [X] Create a stand alone dataset class file to hold the deliciuos config so we can import it into `validated_model.py`
- [X] Solve `load_state_dict` key error when loading the torch model computational graph from Peachy-Sweep
- [X] Plot the knn neighbors for Sierra
- [ ] Plot the embeddings from validation and compare with the embeddings on the final epoch of training (or do what the other teams are doing and plot for embeddings for training data too)
  - [ ] tsne, pca, umap
- [X] Update the `sweeps.yaml` to include encoder architecture (fun fact this is already implemented in the argparser code)
- [ ] Additional slide for Sunflower (at request of Subhamoy and Andres) Show sweep outcomes and best hyperparameters and encoder architecture results from wandb
- [X] Profit

## 09/11/2023
- [ ] Plot the embeddings from validation and compare with the embeddings on the final epoch of training (or do what the other teams are doing and plot for embeddings for training data too)
  - [ ] tsne(done), pca, umap
- [ ] randomize augmentation of values within a neighborhood
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [ ] configure sweeps plot and log more than just loss
    - [ ] [link](https://wandb.ai/site/articles/running-hyperparameter-sweeps-to-pick-the-best-model-using-w-b)
    
- [ ] configure `simclr.py` model to use pretrained base encoder architecture
    - [ ] add wandb.log to training loop
        - [ ] epochs
        - [ ] accuracy
- [ ] finish loading from checkpoint and callback
- [ ] fix densenet101
- [ ] reduce model memory
    - [x] reduce the 64 doubles to 32 floats
- [ ] profit 

## 09/13/23
- [ ] Plot the embeddings from validation and compare with the embeddings on the final epoch of training (or do what the other teams are doing and plot for embeddings for training data too)
  - [X] tsne
  - [ ] pca
  - [ ] umap
- [X] Configure with pre-trained weights

- [ ] configure sweeps plot and log more than just loss
    - [ ] [link](https://wandb.ai/site/articles/running-hyperparameter-sweeps-to-pick-the-best-model-using-w-b)
    
- [X] configure `simclr.py` model to use pretrained base encoder architecture
    - [X] add wandb.log to training loop
        - [X] epochs
        - [ ] accuracy
- [ ] finish loading from checkpoint and callback
- [ ] fix ~~densenet101~~
- [ ] reduce model memory
    - [x] reduce the 64 doubles to 32 floats
- [ ] randomize augmentation of values within a neighborhood
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [ ] profit 

## 9/18/23
- [ ] Plot the embeddings from validation and compare with the embeddings on the final epoch of training (or do what the other teams are doing and plot for embeddings for training data too)
  - [X] tsne make markers smaller, and outline points
  - [x] pca
  - [ ] umap
  - [ ] configure sweeps plot and log more than just loss
    - [ ] [link](https://wandb.ai/site/articles/running-hyperparameter-sweeps-to-pick-the-best-model-using-w-b)
  - [X] (we understand now) fix bug regarding 5x less images being plotted based on number of images in config 
- [ ] finish loading from checkpoint and callback
- [ ] fix ~~densenet101~~
- [ ] reduce model memory
- [ ] randomize augmentation of values within a neighborhood
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [x] profit 

## 9/20/23
- [ ] Plot the embeddings from validation and compare with the embeddings on the final epoch of training (or do what the other teams are doing and plot for embeddings for training data too)
  - [ ] umap
  - [ ] configure sweeps plot and log more than just loss
    - [ ] [link](https://wandb.ai/site/articles/running-hyperparameter-sweeps-to-pick-the-best-model-using-w-b)
- [ ] finish loading from checkpoint and callback
- [ ] fix ~~densenet101~~
- [ ] reduce model memory
- [ ] randomize augmentation of values within a neighborhood
    - [ ] Brighten
    - [ ] Translate
    - [ ] Zoom
    - [ ] Blur
- [ ] Plot only one channel for visualizations in ``vis_utils.py``
- [ ] profit 