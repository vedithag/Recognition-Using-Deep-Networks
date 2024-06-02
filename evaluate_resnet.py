# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/03/2024

# Extension: Visualizing Convolutional Filters and Applying Filter to Image using ResNet-18 

import torch  # Importing the PyTorch library
import torchvision.models as models  # Importing pre-trained models from torchvision
import matplotlib.pyplot as plt  # Importing the matplotlib library for visualization
import cv2  # Importing OpenCV library for image processing
import numpy as np  # Importing NumPy library for numerical computations
from torchvision import transforms, datasets  # Importing torchvision for dataset handling and transformations

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)  # Loading the pre-trained ResNet-18 model
model.eval()  # Set the model to evaluation mode
# Access the weights of the first convolutional layer
weights = model.conv1.weight.data.cpu().numpy()  # Extracting the weights of the first convolutional layer

# Visualize the filters
fig, axes = plt.subplots(8, 8, figsize=(12, 12))  # Creating subplots for visualization
for i, ax in enumerate(axes.flat):  # Iterating through the filters
    if i < weights.shape[0]:  # Checking if the index is within the range of available filters
        ax.imshow(weights[i, 0], cmap='viridis')  # Visualizing the first channel of each filter
        ax.axis('off')  # Turning off axis labels
plt.show()  # Displaying the visualization

# Load and preprocess an image
transform = transforms.Compose([  # Define a sequence of transformations
    transforms.Resize((224, 224)),  # Resizing the image to fit ResNet-18 input size
    transforms.ToTensor(),  # Converting the image to a PyTorch tensor
])

# Load an example image from CIFAR10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # Loading CIFAR10 dataset
image, _ = dataset[0]  # Retrieving the first image in the dataset

# Prepare the image for OpenCV
image_np = image.permute(1, 2, 0).numpy()  # Converting the image to NumPy array and changing dimensions
image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)  # Converting RGB image to BGR (OpenCV format)

# Apply one of the filters to the image
filter_idx = 0  # Choosing an example filter (e.g., first filter)
filter_kernel = weights[filter_idx, 0]  # Extracting the kernel of the selected filter
filtered_image = cv2.filter2D(src=image_np, ddepth=-1, kernel=filter_kernel)  # Applying the filter to the image

# Show the original and filtered images
plt.figure(figsize=(10, 5))  # Creating a new figure for plotting
plt.subplot(1, 2, 1)  # Creating subplots for original image
plt.imshow(np.clip(image_np, 0, 1))  # Displaying the original image, clipped to [0,1] range
plt.title('Original Image')  # Setting the title for the subplot
plt.axis('off')  # Turning off axis labels

plt.subplot(1, 2, 2)  # Creating subplots for filtered image
plt.imshow(filtered_image, cmap='gray')  # Displaying the filtered image
plt.title(f'Filtered Image - Filter {filter_idx}')  # Setting the title for the subplot
plt.axis('off')  # Turning off axis labels

plt.show()  # Displaying the plots
