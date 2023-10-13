#!/bin/bash

# Google Drive URL of the file
FILE_URL='https://drive.google.com/uc?id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY'
#FILE_URL='https://drive.google.com/file/d/1X8RSekrAOryVGSOeC18NOctfJCQrZCUa'

# Get the root directory path where .git lives
# root_directory="$(git rev-parse --show-toplevel)"
# echo ${root_directory}
# DATA_DIR="$root_directory/data"
DATA_DIR="/d0/euv/aia/preprocessed/AIA_211_193_171/AIA_211_193_171_256x256"
# Script for download the file from Google Drive
# gdrive_download () {
#   CONFIRM=$(curl -sc /tmp/gcookie "${FILE_URL}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
#   curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY" -o ${DATA_DIR}/download.tar.gz   
#   #curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=1X8RSekrAOryVGSOeC18NOctfJCQrZCUa" -o ${DATA_DIR}/download.tar.gz   
# }

# Call the gdrive_download if the file does not exist
# if [ ! -f "${DATA_DIR}/download.tar.gz" ]; then
#     gdrive_download
# fi

# Make a directory to store the text file containing all the paths of the .jpg files if it doesn't already exist
if [ ! -d "${DATA_DIR}/train_val_simclr" ]; then
    mkdir ${DATA_DIR}/train_val_simclr
fi

# Output file path
output_file="${DATA_DIR}/train_val_simclr/tot_full_path_files.txt"
# If output_file does not exist, create it
if [ ! -f "${output_file}" ]; then
    #tar --wildcards --force-local -tzvf ${DATA_DIR}/download.tar.gz '*.jpg'|awk '{print $NF}' > ${output_file}
    find ${DATA_DIR} -type f -name "*.jpg" | awk -F/'{print $NF}' > ${output_file}
fi


# If the download.tar.gz has not been extracted, extract it
# first_file=$(head -n 1 ${output_file})
# if [ ! -f "${DATA_DIR}/${first_file}" ]; then
#     tar --force-local -xzvf ${DATA_DIR}/download.tar.gz -C ${DATA_DIR}
# fi

# Remove .gz file
#rm ${DATA_DIR}/download.tar.gz