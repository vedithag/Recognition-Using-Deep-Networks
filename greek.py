# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/03/2024

# Task 3: Transfer learning to the greek letters

# import statements
import sys  # Import the sys module for system-specific parameters and functions
import torch  # Import the torch module, which is the main PyTorch package
from torch import nn  # Import neural network modules from PyTorch
import torch.nn.functional as F  # Import functional interface for common operations on neural networks
from torch.utils.data import Dataset  # Import Dataset class from torch.utils.data module
from torch.utils.data import DataLoader  # Import DataLoader class from torch.utils.data module
import torchvision  # Import the torchvision module, which consists of popular datasets, model architectures, and common image transformations for computer vision tasks
from torchvision import datasets  # Import standard datasets such as MNIST from torchvision
from torchvision.transforms import ToTensor  # Import ToTensor transformation which converts PIL image or numpy.ndarray to tensor
import torchvision.models as models  # Import pre-trained models from torchvision.models
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting functionalities

from task1 import MyNetwork  # Import MyNetwork class from task1 module

# Define class names for Greek letters
class_names = ['alpha', 'beta', 'gamma', 'delta', 'lambda', 'phi']

# Define transformation for Greek dataset
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # Convert RGB image to grayscale
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # Apply affine transformation
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        # Center crop the image
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        # Invert the image
        return torchvision.transforms.functional.invert(x)

# Load and transform the Greek letters from the specified dataset
def loadData(training_set_path):
    # DataLoader for the Greek dataset
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              GreekTransform(),
                                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])),
        batch_size=5,
        shuffle=True)
    return greek_train

# Function to train the network and plot the training result
def train_network(train_dataloader, model, loss_fn, optimizer, epochs):
    train_losses = []  # Initialize an empty list to store training losses
    train_counter = []  # Initialize an empty list to store training steps
    
    # Call the train_loop for each epoch and store the loss in an array
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")  # Print current epoch number
        train_loop(train_dataloader, model, loss_fn, optimizer, train_losses, train_counter, t)
    print("Done!")  # Print message when training is completed

    # Plot the training error
    fig = plt.figure()  # Create a new figure
    plt.plot(train_counter, train_losses, color='blue')  # Plot training losses
    plt.legend(['Train Loss'], loc='upper right')  # Add legend
    plt.xlabel('number of training examples seen')  # Label x-axis
    plt.ylabel('negative log likelihood loss')  # Label y-axis
    plt.show()  # Display the plot

    return

# The train_loop for one epoch, the statistics are saved in losses and counter.
def train_loop(dataloader, model, loss_fn, optimizer, losses, counter, epoch_idx):
    # Set the mode of the model being training, this will affect the dropout layer
    model.train()  # Set model to training mode
    size = len(dataloader.dataset)  # Get size of the dataset
    correct = 0  # Initialize counter for correct predictions
    for batch_idx, (data, target) in enumerate(dataloader):  # Iterate over batches in the dataset
        # Forward pass
        pred = model(data)  # Perform forward pass through the model
        loss = loss_fn(pred, target)  # Calculate loss
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()  # Count correct predictions
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        # Log in the terminal for each 2 batches (10 images)
        if batch_idx % 2 == 0:
            loss, current = loss.item(), (batch_idx+1) * len(data)  # Get loss and current step
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Print loss
            # Save the loss data and its index for final plot
            losses.append(loss)
            counter.append(batch_idx * len(data) + epoch_idx * size)
    # Calculate the accuracy of each training epoch
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")  # Print accuracy
        
# Test with the new Greek letters
def test_on_new(dataloader, model):
    # Set the mode to evaluation
    model.eval()  # Set model to evaluation mode
    # Create the plot window
    size = len(dataloader.dataset)  # Get size of the dataset
    figure = plt.figure(figsize=(9, 9))  # Create a figure with specified size
    cols, rows = 3, (size + 2) // 3  # Calculate number of columns and rows for subplots
    # Main loop which tests an image and plots it with the prediction result
    with torch.no_grad():  # Disable gradient calculation
        for sub_idx, (img, target) in enumerate(dataloader.dataset):  # Iterate over images in dataset
            # Assuming img shape is (C, H, W), need to unsqueeze to (1, C, H, W) for model
            img_unsqueezed = img.unsqueeze(0)  # Add a dimension for batch
            pred = model(img_unsqueezed)  # Get model prediction
            pred_label = class_names[pred.argmax(1).item()]  # Get predicted label
            target_label = class_names[target]  # Get actual label
            
            # Plot the subplot
            figure.add_subplot(rows, cols, sub_idx + 1)  # Add subplot to the figure
            plt.title(f"Pred: {pred_label}, Target: {target_label}")  # Set title for subplot
            plt.axis("off")  # Turn off axis
            plt.imshow(img.squeeze(), cmap="gray")  # Display image
    plt.show()  # Show the plot

# Main function
def main(argv):
    # Handle argv
    commands = ['original', 'add', 'extend']  # Define valid commands
    if len(argv) < 2:
        mode = 'original'  # Set default mode if not provided
    else:
        mode = argv[1]  # Get mode from command line argument
    if mode not in commands:
        print("Command not accepted")  # Print error message for invalid command
        exit(-1)  # Exit program with error code

    # Make sure the model is repeatable
    random_seed = 47  # Set random seed for reproducibility
    torch.manual_seed(random_seed)  # Set random seed for PyTorch
    torch.backends.cudnn.enabled = False  # Disable cuDNN for deterministic behavior

    # Set the settings for the training
    learning_rate = 1e-2  # Set learning rate for optimization
    epochs = 40  # Set number of epochs for training
    
    # Load the trained model
    model = torch.load('model.pth')  # Load pre-trained model
    # Freeze the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False  # Set requires_grad to False for all parameters
    # Replace the last layer with a new 3-nodes layer
    if mode != 'extend': 
        model.fc2 = nn.Linear(50, 3)  # Replace last fully connected layer with 3 output nodes
    else:
        model.fc2 = nn.Linear(50, 6)  # Replace last fully connected layer with 6 output nodes

    # Decide which train and test set to use according to the mode
    if mode == 'original':
        greek_train_path = './greek_letters/greek_train'  # Set path for original Greek train dataset
        greek_test_path = './greek_letters/greek_test'  # Set path for original Greek test dataset
    elif mode == 'add':
        greek_train_path = './greek_letters/additional_greek_train_v1'  # Set path for additional Greek train dataset
        greek_test_path = './greek_letters/greek_test'  # Set path for original Greek test dataset
    else:
        greek_train_path = './greek_letters/extended_greek_train'  # Set path for extended Greek train dataset
        greek_test_path = './greek_letters/extended_greek_test'  # Set path for extended Greek test dataset

    # Load the training data and test data
    greek_train_loader = loadData(greek_train_path)  # Load training data
    greek_test_loader = loadData(greek_test_path)  # Load test data

     # Loss function
    loss_fn = nn.NLLLoss()  # Define loss function (Negative Log Likelihood Loss)
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Define optimizer (Stochastic Gradient Descent)
    # Start training
    train_network(greek_train_loader, model, loss_fn, optimizer, epochs)  # Train the network
    # Print out the model (for report)
    print(model)  # Print the model architecture
    # Start testing
    test_on_new(greek_test_loader, model)  # Test the model on new Greek letters

    return

if __name__ == "__main__":
    main(sys.argv)  # Call main function with command line arguments
