import json
import numpy as np
import torch
import cv2
import os

# Function to load images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

# Function to convert list of images to numpy array
def images_to_numpy_array(images):
    return np.array(images)

# Load the JSON file
with open(r'C:\Users\sreej\OneDrive\Documents\Instant-NGP-for-RTX-3000-and-4000\Gluedataset\transforms.json') as f:
    data = json.load(f)

# Extract transform_matrix from each frame and store them in a list
transform_matrices = [np.array(frame['transform_matrix']) for frame in data['frames']]

# Convert the list of matrices into a numpy array
transform_matrices_array = np.array(transform_matrices, dtype=np.float32) # Specify dtype here

# Convert to float tensor
poses_tensor = torch.tensor(transform_matrices_array, dtype=torch.float32)

# Extract focal length values
fl_x = data['fl_x']
fl_y = data['fl_y']

# Calculate average focal length
focal_length_avg = (fl_x + fl_y) / 2

# Convert to float tensor
focal_tensor = torch.tensor(focal_length_avg, dtype=torch.float64)  # Specify dtype here

# Replace 'folder_path' with the path to your folder containing images
import os
import numpy as np
from PIL import Image

def resize_images(folder_path, factor=10):
    images = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                new_size = (width // factor, height // factor)
                img = img.resize(new_size)
                img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # Normalize and specify dtype as float32
                images.append(img)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
    return np.array(images)

folder_path = r"C:\Users\sreej\OneDrive\Documents\Instant-NGP-for-RTX-3000-and-4000\Gluedataset\images"
resized_images = resize_images(folder_path)

print("Shape of resized images array:", resized_images.shape)

np.savez('gluedata.npz', images=resized_images,
         poses=poses_tensor.numpy(), focal=focal_tensor.numpy())
print("Data saved to: gluedata.npz")
