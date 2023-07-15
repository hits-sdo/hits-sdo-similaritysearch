#!/bin/bash

# Google Drive URL of the file
FILE_URL='https://drive.google.com/uc?id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY'

# Script for download the file from Google Drive
gdrive_download () {
  CONFIRM=$(curl -sc /tmp/gcookie "${FILE_URL}" | grep -o 'confirm=[^&]*' | sed 's/confirm=//')
  curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=1DMIatOmA4XcoWeW0oAUkZujx8YrhLkpY" -o ../../../data/dl.gz   
}

# Call the gdrive_download
gdrive_download

# Unzip the file
tar -xzvf ../../../data/dl.gz

# Remove .gz file
rm ../../../data/dl.gz

# traverse the file directory tree find all leaf node files and append paths of leaf node files to a .txt


# C:\Projects\HITS-GIBS\Team_Sunflower_Phase_2\hits-sdo-similaritysearch\data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles\20100601_000008_aia_211_193_171_tile_256_256.jpg
# data\AIA211_193_171_Miniset\20100601_000008_aia_211_193_171\tiles\20100601_000008_aia_211_193_171_tile_256_256.jpg