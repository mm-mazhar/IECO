# -*- coding: utf-8 -*-
# """
# check_invalid_labels.py
# Created on Oct Sept 02, 2024
# """
import os
import numpy as np

def check_invalid_labels_in_seg_files(directory):
    # List to store files with invalid labels
    files_with_invalid_labels = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".seg"):
            file_path = os.path.join(directory, filename)
            # Load the .seg file
            labels = np.loadtxt(file_path, dtype=int)
            # Check if any label is not 0 or 1
            if np.any((labels != 0) & (labels != 1)):
                files_with_invalid_labels.append(filename)

    # Report the results
    if files_with_invalid_labels:
        print(f"Files with labels other than 0 or 1: {files_with_invalid_labels}")
    else:
        print("No .seg files with labels other than 0 or 1 found.")

# Usage example: replace 'your_directory_path' with the actual path where your .seg files are located.
directory_path = './points_label/'
check_invalid_labels_in_seg_files(directory_path)
