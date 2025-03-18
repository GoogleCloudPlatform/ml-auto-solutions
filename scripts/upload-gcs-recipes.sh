#!/usr/bin/env bash
# upload_to_gcs.sh
# Script to upload files from specified folders to a GCS bucket
# Prerequisites: gsutil must be installed and configured

# Set your GCS bucket name and prefix
BUCKET_NAME="yujunzou-dev-supercomputer-testing"
BUCKET_PREFIX="internal_gpu_recipes"

# Default directories
VALUES_DIR="./values"
RECIPES_DIR="./recipes"

# Function to check if directory exists
check_directory() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "Error: Directory '$dir' does not exist."
        echo "Current working directory is: $(pwd)"
        echo "Available directories and files:"
        ls -la
        return 1
    fi
    return 0
}

# Function to upload a directory to GCS
upload_directory_to_gcs() {
    local local_dir=$1
    local gcs_prefix=$2

    # Check if directory exists before proceeding
    if ! check_directory "$local_dir"; then
        echo "Skipping upload for $local_dir"
        return 1
    fi
    
    echo "Uploading files from $local_dir to gs://$BUCKET_NAME/$BUCKET_PREFIX/$gcs_prefix/"
    
    # Ensure the directory ends with a slash
    if [[ ! "$local_dir" == */ ]]; then
        local_dir="${local_dir}/"
    fi
    
    # Find all files in the directory and upload each one
    find "$local_dir" -type f | while read local_file; do
        # Get the relative path by removing the local directory prefix
        relative_path="${local_file#$local_dir}"
        
        # Construct the GCS path
        gcs_path="$BUCKET_PREFIX/$gcs_prefix/$relative_path"
        
        # Upload the file
        gsutil cp "$local_file" "gs://$BUCKET_NAME/$gcs_path"
        
        echo "Uploaded: $local_file → gs://$BUCKET_NAME/$gcs_path"
    done
    
    echo "Upload completed for $local_dir"
    return 0
}

# Display current working directory and available files/folders
echo "Current working directory: $(pwd)"
echo "Available files and folders:"
ls -la

# Upload the values folder
upload_directory_to_gcs "$VALUES_DIR" "values"

# Upload the recipes folder
upload_directory_to_gcs "$RECIPES_DIR" "recipes"

echo "All uploads completed!"