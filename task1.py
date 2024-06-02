# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/03/2024

# Task 1: Build and train a network to recognize digits

# Import necessary libraries
import sys  # System-specific parameters and functions
import torch  # PyTorch deep learning framework
from torch import nn  # Neural network module of PyTorch
import torch.nn.functional as F  # Functional interface for neural network operations
from torch.utils.data import Dataset, DataLoader  # Dataset and data loader utilities for PyTorch
from torchvision import datasets  # Datasets for PyTorch
from torchvision.transforms import ToTensor  # Transform PIL Image or numpy.ndarray to tensor
import torchvision.models as models  # Pre-trained models for computer vision tasks
import matplotlib.pyplot as plt  # Plotting library for Python

from task1 import MyNetwork  # Importing custom neural network model from task1

# Load digit MNIST
def loadData():
    # Get the training data
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # Get the test data
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return (training_data, test_data)

# Helper function that plots the first 6 images from the MNIST digits
def showExamples(training_data):
    figure = plt.figure(figsize=(8, 6))  # 8x6 inches window
    cols, rows = 3, 2
    for i in range(cols * rows):
        img, label = training_data[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# The structure of the model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # The stack of convolution layers and the followed relu and max pooling layers
        self.my_stack = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # Convolution layer 1, 28*28 to 10*24*24
            nn.MaxPool2d(2),  # 24*24 to 10*12*12
            nn.ReLU(),  # ReLU layer
            nn.Conv2d(10, 20, kernel_size=5),  # 10*12*12 to 20*8*8
            nn.Dropout(0.5),
            nn.MaxPool2d(2),  # 20*8*8 to 20*4*4
            nn.ReLU()
        )
        # Fully connected layers for classification
        self.fc1 = nn.Linear(320, 50)  # 20*4*4 = 320
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass for the network
    def forward(self, x):
        x = self.my_stack(x)  # The convolution layers
        x = x.view(-1, 320)  # Flatten according to the size and channels, 320 = 20x4x4
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU
        x = F.log_softmax(self.fc2(x), dim=1)  # Log_softmax after the fc2
        return x

# The function that trains the network and plots the result of the training and testing
def train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    # Holder for the result of each epoch
    train_losses = []  # List to store training losses
    train_counter = []  # List to store number of training examples seen
    test_losses = []  # List to store test losses
    test_counter = [i * len(train_dataloader.dataset) for i in range(epochs)]  # List to store number of examples seen in each test epoch

    # Call the train_loop and test_loop for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, train_losses, train_counter, t)
        test_loop(test_dataloader, model, loss_fn, test_losses)
    print("Done!")

    # Plot the training and testing result
    fig = plt.figure()  # Create a new figure
    plt.plot(train_counter, train_losses, color='blue')  # Plot training loss
    plt.scatter(test_counter, test_losses, color='red')  # Plot test loss
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # Add legend
    plt.xlabel('number of training examples')  # Set x-axis label
    plt.ylabel('negative log loss')  # Set y-axis label
    plt.show()  # Display the plot

    return

# The train_loop for one epoch, the statistics are saved in losses and counter.
def train_loop(dataloader, model, loss_fn, optimizer, losses, counter, epoch_idx):
    # Set the mode of the model being training, this will affect the dropout layer
    model.train()
    size = len(dataloader.dataset)  # Get the size of the dataset
    for batch_idx, (data, target) in enumerate(dataloader):
        # Forward pass
        pred = model(data)  # Get predictions from the model
        loss = loss_fn(pred, target)  # Compute the loss
        # Backward pass
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update parameters
        # Log in the terminal for each 10 batches
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Print loss
            losses.append(loss)  # Append loss to the list
            counter.append(batch_idx * len(data) + epoch_idx * size)  # Append counter value

# The test_loop for one epoch, the statistics are saved in losses.
def test_loop(dataloader, model, loss_fn, losses):
    # Set the mode of the model being testing, this will affect the dropout layer
    model.eval()
    # Variables for accuracy computing
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Get the test loss for each batch
    with torch.no_grad():
        for data, target in dataloader:
            pred = model(data)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    # Calculate and log the accuracy of test
    test_loss /= num_batches
    losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Main function
def main(argv):
    # Task 1-b
    random_seed = 47  # Make the model repeatable
    torch.manual_seed(random_seed)  # Set random seed for reproducibility
    torch.backends.cudnn.enabled = False  # Disable cuDNN for reproducibility

    # Set the settings for the training
    learning_rate = 1e-2  # Learning rate for optimizer
    batch_size = 64  # Batch size for training
    epochs = 5  # Number of training epochs

    # Load training and test data
    training_data, test_data = loadData()
    # Uncomment to show the first six images.
    showExamples(training_data)

    # Create dataloaders
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    # Get an instance of the network
    model = MyNetwork()

    # Loss function
    loss_fn = nn.NLLLoss()  # Didn't choose CrossEntropyLoss because we already did the logsoftmax in the model
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # Start training
    train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
    # Save the trained network
    torch.save(model, 'model.pth')

    return

if __name__ == "__main__":
    main(sys.argv)
