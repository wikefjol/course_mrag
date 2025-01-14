import os
import hashlib
import json
import time
import datetime
import shutil


def compute_folder_hash(folder_path: str) -> str:
    """
    Calculates a hash value based on the file contents of a folder.
    If none of the files have changed, the hash remains the same.
    This helps us skip unnecessary vector store rebuilding.
    """
    md5 = hashlib.md5()
    
    # Walk through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            
            # We only consider certain file types
            # (In practice, you might consider only pdf, pptx, txt, etc.)
            if file_name.lower().endswith(('.pdf', '.txt', '.pptx')):
                with open(file_path, 'rb') as f:
                    # Update the hash
                    data = f.read()
                    md5.update(data)
    
    return md5.hexdigest()

def delete_folder(folder_path: str):
    """
    Deletes the specified folder after user confirmation.
    """
    folder_path = os.path.abspath(folder_path)

    if os.path.exists(folder_path):
        confirm = input(f"Are you sure you want to delete the folder: {folder_path}? (yes/no): ")
        if confirm.lower() == "yes":
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print("Deletion cancelled.")
    else:
        print(f"Folder does not exist: {folder_path}")