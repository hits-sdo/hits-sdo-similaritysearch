#!/bin/bash
DATA_DIR="../../../data"
ARG1=$1
echo $ARG1
ARG2=$2
# Get the root directory path where .git lives
root_directory="$(git rev-parse --show-toplevel)"
ARG1=$root_directory

# Function to check if a directory is empty
is_directory_empty() {
    if [ -z "$(ls -A $ARG1)" ]; then
        return 0
    else
        return 1
    fi
}



# Function to filter the files and find the leaf node .jpg files
find_leaf_node_jpg_files() {
    local directory=$ARG1
    echo $directory
    local output_file=`echo $2`

    # Check if the directory is empty
    if is_directory_empty "$directory"; then
        return
    fi

    # Find .jpg files in the directory and its subdirectories
    find "$directory" -type f -name "*.jpg" | while read -r file; do
        # Check if the file is in a leaf node directory
        if is_directory_empty "$(dirname "$file")"; then
            #Get the relative path to the .git directory
            relative_path="${file#$root_directory/}"
            echo "$relative_path" >> "$output_file"
        fi
    done
}

# Output file path
output_file="${DATA_DIR}/train_val_simclr/tot_full_path_files.txt"

# Find and filter the leaf node .jpg files
find_leaf_node_jpg_files "$root_directory" "$output_file"

# Print the number of files found
file_count=$(wc -l < "$output_file")
echo "$file_count .jpg files found and saved to $output_file"

