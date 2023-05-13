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
        

## Consideration For Next Meeting
- Random augmentations on a tile saved in json files after augmentation (parallel arrays)
    -Naming convention - same name, different type augmentation (buddy naming, jason + tile thats already augmented)
- OR
- Random augmentations in the data loader, by taking advatage of the composed transform function in pytorch
    - If so, refactor random augs from team yellow, or load the json file that were generated
