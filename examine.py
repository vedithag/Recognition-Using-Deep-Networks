# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/03/2024

# Task 2: Examine your networks

# Import necessary libraries
import sys  # System-specific parameters and functions
import torch  # PyTorch deep learning framework
import numpy as np  # Numerical computing library
import cv2 as cv  # OpenCV computer vision library
from torch import nn  # Neural network module of PyTorch
import torch.nn.functional as F  # Functional interface for neural network operations
from torchvision import datasets  # Datasets and data loaders for PyTorch
from torchvision.transforms import ToTensor  # Transform PIL Image or numpy.ndarray to tensor
import matplotlib.pyplot as plt  # Plotting library for Python

from task1 import MyNetwork  # Importing custom neural network model from task1

# Main function
def main(argv):

    # Load the model and set the mode to evaluation
    model = torch.load('model.pth')
    model.eval()

    # Print out the model in the terminal
    print(model)

    # Get the weights of the first convolution layer (which is the first of the my_stack sequential)
    weights = model.my_stack[0].weight

    # Visualize the weights
    figure=plt.figure(figsize=(9,8))  # Set figure size to 8 inches
    cols, rows = 4, 3  # Define subplot layout
    with torch.no_grad():
        for i in range(weights.shape[0]):
            weight = weights[i,0]
            figure.add_subplot(rows, cols, i+1)
            plt.title(f"Filter: {i}")
            plt.axis("off")
            plt.imshow(weight.squeeze())
    plt.show()

    # Load the first image from the training data
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # [0] is the first example, [0] is for the first entry(data), [0] is to make [1,28,28] to [28,28]
    img = training_data[0][0][0] 
    # Convert to array so that it can be processed by OpenCV
    img_array = img.numpy()

    # Loop for applying the filters and then plot the result
    filter_figure = plt.figure(figsize=(9,8))  # Set figure size to 8 inches
    cols, rows = 4, 5  # Define subplot layout
    with torch.no_grad():
        for i in range(weights.shape[0]):
            # Get the filter and then use filter2D to apply it on the image
            filter = weights[i,0]
            filter_array = filter.numpy()
            filter_img = cv.filter2D(img_array, -1, filter_array)
            # Subplot the filtering itself
            filter_figure.add_subplot(rows, cols, 2*i+1)
            plt.title(f"Filter: {i}")
            plt.axis("off")
            plt.imshow(filter.squeeze(), cmap="gray")
            # Subplot the filtered result
            filter_figure.add_subplot(rows, cols, 2*i+2)
            plt.title(f"Result: {i}")
            plt.axis("off")
            plt.imshow(filter_img.squeeze(), cmap="gray")
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
