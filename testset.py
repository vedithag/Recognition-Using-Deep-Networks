# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/03/2024

# Task 1E and Task 1F: Build and train a network to recognize digits


# Import necessary libraries
import sys  # System-specific parameters and functions
import torch  # PyTorch deep learning framework
import numpy as np  # NumPy library for numerical operations
import cv2 as cv  # OpenCV library for computer vision tasks
from torch import nn  # Neural network module of PyTorch
import torch.nn.functional as F  # Functional interface for neural network operations
from torchvision import datasets  # Datasets for PyTorch
from torchvision.transforms import ToTensor  # Transform PIL Image or numpy.ndarray to tensor
import matplotlib.pyplot as plt  # Plotting library for Python

# Import the model structure
from task1 import MyNetwork  # Custom neural network model

# Task F: Test the model on the examples in the dataset
def testOnDataset(model):
    # Load the test set
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Start testing and store the predictions in the list for later plotting
    predictions = []
    with torch.no_grad():
        # Test the first 10 images
        for i in range(10):
            data, target = test_data[i]
            print()
            pred = model(data.unsqueeze(0))  # Add batch dimension
            predictions.append(pred.argmax(1).item())
            pred_array = pred.numpy()[0]  # Convert tensor to numpy array
            # Print out the 10 values of each pred_array
            for value in pred_array:
                print(f"{value:<6.2f}", end=" ")
            print()
            # The real prediction is the index of the largest value
            print(f"Prediction: {pred.argmax(1).item()}, Target: {target}")
    
    # After testing the first 10 examples, plot the first 9 of them
    figure = plt.figure(figsize=(9, 12))  # 9x12 inches 
    cols, rows = 3, 3
    for i in range(9):
        img, target = test_data[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(f"Prediction: {predictions[i]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# Task G: Test the model on new digits
def test_On_New(model):
    test_data = []
    # Pre-process the image so that it is similar to the digits from the datasets
    for i in range(10):
        # Read the image
        img = cv.imread('new_num/' + str(i) + '.jpg')
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # Convert to grayscale
        invert = cv.bitwise_not(gray)  # Invert the image
        invert = invert / 255  # Normalize the intensity to [0, 1]
        addDim = invert[np.newaxis, :, :]  # Add batch dimension
        tensor_trans = torch.tensor(addDim, dtype=torch.float32)  # Convert to PyTorch tensor
        test_data.append(tensor_trans)
    
    # Start testing and store the predictions in the list for later plotting
    predictions = []
    with torch.no_grad():
        for i in range(10):
            data = test_data[i]
            print()
            pred = model(data.unsqueeze(0))  # Add batch dimension
            predictions.append(pred.argmax(1).item())
            pred_array = pred.numpy()[0]  # Convert tensor to numpy array
            for value in pred_array:
                print(f"{value:<6.2f}", end=" ")  # Print the prediction values
            print()
            # The real prediction is the index of the largest value
            print(f"Prediction: {pred.argmax(1).item()}, Target: {i}")
            
    # After testing the first 10 examples, plot them
    figure = plt.figure(figsize=(9, 12))  # 9x12 inches 
    cols, rows = 3, 4
    for i in range(10):
        img = test_data[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(f"Prediction: {predictions[i]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    

def main(argv):
    
    # Load the saved model and set the mode to evaluation
    model = torch.load('model.pth')
    model.eval()

    # Task F: Test on dataset
    testOnDataset(model)
    # Task G: Test on new digits
    test_On_New(model)

    return

if __name__ == "__main__":
    main(sys.argv)
