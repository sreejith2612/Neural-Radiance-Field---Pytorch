import numpy as np

data = np.load('gluedata.npz')

# Extract images, poses, and focal length
images = data['images']
poses = data['poses']
focal = data['focal']

# Print shapes
print("Shape of images:", images.shape if images is not None else None)
print("Shape of poses:", poses.shape)
print("Shape of focal:", focal.shape)

# Print datatype of each element
if images is not None:
    print("Datatype of elements in images:", images.dtype)
print("Datatype of elements in poses:", poses.dtype)
print("Datatype of elements in focal:", focal.dtype)

