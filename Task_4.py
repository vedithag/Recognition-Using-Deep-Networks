# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/04/2024

# Task 4: Designing Our Experiment with MNIST Fashion data set 

import numpy as np  # Import numpy library for numerical operations
import time  # Import time library for time-related functions
from keras.datasets import fashion_mnist  # Import fashion_mnist dataset from keras
from keras.models import Sequential  # Import Sequential model from keras for building a sequential neural network
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Import layers from keras for building the model
from keras.optimizers import Adam  # Import Adam optimizer from keras for optimization
from keras.utils import to_categorical  # Import to_categorical from keras for one-hot encoding
from sklearn.model_selection import train_test_split  # Import train_test_split from sklearn for splitting dataset
import matplotlib.pyplot as plt  # Import matplotlib library for plotting graphs

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # Load data into training and testing sets

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255  # Reshape and normalize training data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # Reshape and normalize testing data
y_train = to_categorical(y_train, 10)  # Perform one-hot encoding on training labels
y_test = to_categorical(y_test, 10)  # Perform one-hot encoding on testing labels

# Split validation data from training data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)  # Split training data into training and validation sets

# Define ranges for experimentation
num_conv_layers_range = [1, 2, 3]  # Number of convolutional layers to explore
num_filters_range = [16, 32, 64]  # Number of filters to explore
dense_nodes_range = [128, 256, 512]  # Number of dense nodes to explore
dropout_rate_range = [0.2, 0.3, 0.4, 0.5]  # Dropout rates to explore

# Results dictionary to store performance metrics
results = []

# Linear search strategy
for num_conv_layers in num_conv_layers_range:  # Loop over number of convolutional layers
    for num_filters in num_filters_range:  # Loop over number of filters
        for dense_nodes in dense_nodes_range:  # Loop over number of dense nodes
            for dropout_rate in dropout_rate_range:  # Loop over dropout rates
                # Build the model
                model = Sequential()  # Initialize a Sequential model
                # Add convolutional layers
                for _ in range(num_conv_layers):  # Loop to add convolutional layers
                    model.add(Conv2D(num_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Add convolutional layer
                    model.add(MaxPooling2D((2, 2)))  # Add max pooling layer
                model.add(Flatten())  # Flatten the output for fully connected layers
                # Add fully connected layers
                model.add(Dense(dense_nodes, activation='relu'))  # Add dense layer with specified number of nodes
                model.add(Dropout(dropout_rate))  # Add dropout layer with specified dropout rate
                model.add(Dense(10, activation='softmax'))  # Add output layer with softmax activation
                
                model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])  # Compile the model
                
                # Train the model
                print(f"Training model with: Num Conv Layers: {num_conv_layers}, Num Filters: {num_filters}, Dense Nodes: {dense_nodes}, Dropout Rate: {dropout_rate}")
                history = model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0, validation_data=(x_val, y_val))  # Train the model
                
                # Print results for each epoch
                print("Results for each epoch:")
                for epoch, acc in enumerate(history.history['val_accuracy'], 1):  # Loop over epochs
                    print(f"Epoch {epoch}: Validation Accuracy: {acc:.4f}")  # Print validation accuracy
                
                # Evaluate the model
                accuracy = model.evaluate(x_val, y_val, verbose=0)[1]  # Evaluate the model on validation data
                
                # Record the results
                results.append({
                    'Num Conv Layers': num_conv_layers,
                    'Num Filters': num_filters,
                    'Dense Nodes': dense_nodes,
                    'Dropout Rate': dropout_rate,
                    'Accuracy': accuracy,
                    'History': history
                })

# Plot training and validation accuracy for each variation
for i, result in enumerate(results):  # Loop over results
    plt.figure(figsize=(8, 6))  # Create a new figure for plotting
    plt.plot(result['History'].history['accuracy'], label='Training Accuracy')  # Plot training accuracy
    plt.plot(result['History'].history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
    plt.title(f"Variation {i+1} - Training and Validation Accuracy")  # Set title for the plot
    plt.xlabel('Epoch')  # Set label for x-axis
    plt.ylabel('Accuracy')  # Set label for y-axis
    plt.legend()  # Add legend to the plot
    plt.show()  # Show the plot

# Print and analyze results
for i, result in enumerate(results):  # Loop over results
    print(f"Variation {i+1}:")  # Print variation number
    print(result)  # Print result dictionary
    print()  # Print empty line for separation
