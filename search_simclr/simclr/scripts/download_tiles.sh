#!/bin/bash

# Google Drive URL of the file
FILE_URL='https://drive.google.com/uc?id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY'
DATA_DIR="../../../data"
# Script for download the file from Google Drive
gdrive_download () {
  CONFIRM=$(curl -sc /tmp/gcookie "${FILE_URL}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
  curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY" -o ${DATA_DIR}/dl.gz   
}

# Call the gdrive_download
gdrive_download

# Unzip the file
tar -xzvf ${DATA_DIR}/dl.gz -C ${DATA_DIR}

# Remove .gz file
rm ${DATA_DIR}/dl.gz

mkdir ${DATA_DIR}/train_val_simclr
# traverse the file directory tree find all leaf node files and append paths of leaf node files to a .txt
#find ${DATA_DIR}/AIA211_193_171_Miniset/ -type f ! -name '*.json' > ${DATA_DIR}/train_val_simclr/tot_full_path_files.txt

realpath --relative-to=$(pwd) ${DATA_DIR}
find ${DATA_DIR}/AIA211_193_171_Miniset/ -type f ! -name '*.json' -exec realpath --relative-to=$(pwd) {} \; > ${DATA_DIR}/train_val_simclr/tot_full_path_files.txt



# C:\Projects\HITS-GIBS\Team_Sunflower_Phase_2\hits-sdo-similaritysearch\data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles\20100601_000008_aia_211_193_171_tile_256_256.jpg
# data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles\20100601_000008_aia_211_193_171_tile_256_256.jpg


# #!/bin/bash

# # Check if the root directory is provided as an argument
# if [ -z "\$1" ]; then
#     echo "Please provide the root directory path as an argument."
#     exit 1
# fi

# # Function to check if a directory is empty
# is_directory_empty() {
#     if [ -z "$(ls -A "\$1")" ]; then
#         return 0
#     else
#         return 1
#     fi
# }

# # Function to filter the files and find the leaf node .jpg files
# find_leaf_node_jpg_files() {
#     local directory="\$1"
#     local output_file="\$2"

#     # Check if the directory is empty
#     if is_directory_empty "$directory"; then
#         return
#     fi
#
#     # Find .jpg files in the directory and its subdirectories
#     find "$directory" -type f -name "*.jpg" | while read -r file; do
#         # Check if the file is in a leaf node directory
#         if is_directory_empty "$(dirname "$file")"; then
#             echo "$file" >> "$output_file"
#         fi
#     done
# }

# # Main script

# # Get the root directory path from the command-line argument
# root_directory="\$1"

# # Output file path
# output_file="leaf_node_jpg_files.txt"

# # Find and filter the leaf node .jpg files
# find_leaf_node_jpg_files "$root_directory" "$output_file"

# # Print the number of files found
# file_count=$(wc -l < "$output_file")
# echo "$file_count .jpg files found and saved to $output_file"
