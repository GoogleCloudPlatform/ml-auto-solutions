from google.cloud import storage
import os

def download_from_gcs(bucket_name="yujunzou-dev-supercomputer-testing", 
                      gcs_prefix="internal_gpu_recipes",
                      local_dir="/tmp/internal_gpu_recipes"):
    """
    Download files from a GCS bucket to a local directory.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcs_prefix: Directory prefix in GCS to download from
        local_dir: Local directory path to download files to
    """
    print(f"Downloading from gs://{bucket_name}/{gcs_prefix}/ to {local_dir}")
    
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Ensure the GCS prefix ends with '/'
        if not gcs_prefix.endswith('/'):
            gcs_prefix += '/'
        
        # Ensure the local directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # List all blobs in the bucket with the specified prefix
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        
        if not blobs:
            print(f"No files found at gs://{bucket_name}/{gcs_prefix}")
            return 0
        
        print(f"Found {len(blobs)} files/objects to download")
        
        download_count = 0
        for blob in blobs:
            # Skip the directory itself
            if blob.name == gcs_prefix:
                continue
            
            # Get the relative path by removing the GCS prefix
            relative_path = blob.name[len(gcs_prefix):]
            if not relative_path:  # Skip empty paths
                continue
                
            # Construct the local file path
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Create any necessary subdirectories
            local_file_dir = os.path.dirname(local_file_path)
            if local_file_dir:
                os.makedirs(local_file_dir, exist_ok=True)
            
            # Download the file
            blob.download_to_filename(local_file_path)
            download_count += 1
            
            print(f"Downloaded: {blob.name} → {local_file_path}")
        
        print(f"Download completed: {download_count} files downloaded")
        return download_count
        
    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise


def list_files_and_dirs(path):
    """
    Print all files and directories under the specified path.
    
    Args:
        path (str): The directory path to list
    """
    print(f"Contents of {path}:")
    
    # Check if the path exists
    if not os.path.exists(path):
        print(f"The path {path} does not exist.")
        return
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(path):
        # Print the current directory (relative to the starting path)
        rel_path = os.path.relpath(root, path)
        if rel_path == ".":
            print(f"- {root} (directory)")
        else:
            print(f"- {root} (directory)")
        
        # Print all directories in the current directory
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"  - {dir_path} (directory)")
        
        # Print all files in the current directory
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"  - {file_path} (file)")

# Main execution with default parameters - no argparse needed
if __name__ == "__main__":
    # Call with default parameters
    # download_from_gcs()
    list_files_and_dirs("./")
    
    # Alternative: You can also pass custom parameters directly
    # download_from_gcs(
    #     bucket_name="your-bucket-name",
    #     gcs_prefix="your-prefix",
    #     local_dir="/your/local/directory"
    # )

