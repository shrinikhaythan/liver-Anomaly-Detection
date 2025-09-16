import zipfile
import os

def extract_zip(zip_path, extract_to=None):
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to

def count_slices(directory):
    count = 0
    for file in os.listdir(directory):
        if file.endswith(('.png', '.jpg', '.dcm')):
            count += 1
    return count