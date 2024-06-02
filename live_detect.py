# Veditha Gudapati & Tejasri Kasturi 
# CS5330 Project 5: Recognition using Deep Network
# 04/04/2024

# Extension: Detect Digits in Live Video  

import torch  # Importing the PyTorch library
import cv2   # Importing OpenCV library for image processing
import torchvision.transforms as transforms
from task1 import MyNetwork  # Importing your custom network from task1.py

# Load your trained model
model = MyNetwork()  # Initialize your custom network
model = torch.load('model.pth', map_location=torch.device('cpu'))  # Load the trained model from file
model.eval()  # Set the model to evaluation mode


# Define transformations that match training
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((28, 28)),  # Resize images to match training data
    transforms.Grayscale(),  # Ensure grayscale
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalize like MNIST
])

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Optional: Apply adaptive thresholding to emphasize the digits
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert the image if your training data is white on black
    gray = cv2.bitwise_not(gray)
    # Apply transformations
    return transform(gray).unsqueeze(0)  # Add batch dimension

cap = cv2.VideoCapture(2)  

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break  # Exit loop if no frame is captured
    
    processed_frame = preprocess_frame(frame)  # Preprocess the frame
    
    with torch.no_grad():  # Turn off gradients for inference
        output = model(processed_frame)  # Forward pass through the model
        _, predicted = torch.max(output, 1)  # Get the index of the class with the highest probability
        prediction = predicted.item()  # Convert to Python scalar
    
    # Display the prediction on the frame
    cv2.putText(frame, f'Predicted Digit: {prediction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Live Digit Recognition', frame)  # Display the frame with prediction
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
