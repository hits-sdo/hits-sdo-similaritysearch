import os

# User Input Here
# 1. Data Directory (Absolute Path) to save the data references
data_dir = '../../data'
# 2. Image Directory (Relative Path) to save the images
image_dir = 'AIA211_193_171_Miniset'

# Data Directory (Absolute Path) to save the data references
if not os.path.exists(data_dir):
    raise Exception('Data directory does not exist')

target_dir = os.path.join(data_dir, image_dir)
if not os.path.exists(target_dir):
    raise Exception('Images directory does not exist')

# Make folders required for setup, training, and validating
paths = [f'{data_dir}/train_val_simclr', '../../model_weights', '../../visualizations', '../../visualizations/simclr_knn', '../../checkpoints']
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)

# Make tot_full_path_files.txt by looping through all the files in the target directory
with open(os.path.join(data_dir, 'train_val_simclr', 'tot_full_path_files.txt'), 'w') as f:
    for root, dirs, files in os.walk(f'{target_dir}'):
        for file in files:
            if file.endswith('.jpg'):
                # Remove the data_dir path when wrting to the file
                f.write(os.path.join(root, file).replace(data_dir+'\\', '') + '\n')