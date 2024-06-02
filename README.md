# Recognition-Using-Deep-Networks

## Overview
This repository contains the implementation of a project focused on calibration and augmented reality (AR). The project involves analyzing and enhancing video streams in real-time using techniques such as chessboard corner extraction, Harris corner recognition, and camera calibration. The calibrated camera data is then used to accurately project virtual objects onto a 2D video feed, maintaining their correct orientation and position even as the target or camera moves.

## Project Structure
The project is divided into several tasks, each addressing a specific aspect of the system:

### Task 1: Detect and Extract Target Corners
Objective: Detect corners of a chessboard pattern using corner detection algorithms.
Implementation:
Corner Detection: Uses findChessboardCorners from the OpenCV library.
Corner Refinement: Applies cornerSubPix for sub-pixel accuracy.
Draw Detected Corners: Visual representation of detected corners using drawCorners.

### Task 2: Select Calibration Images
Objective: Collect and store pixel and world coordinates of detected corners for camera calibration.
Implementation:
World Coordinates Calculation: Calculate world coordinates for each detected corner.
Storage of Coordinates: Store pixel and world coordinates in vectors.
Adding Data to Lists: Add sets of coordinates to respective lists for multiple calibration images.

### Task 3: Calibrate the Camera
Objective: Estimate intrinsic camera parameters and distortion coefficients.
Implementation:
Calibration Using OpenCV: Uses calibrateCamera to estimate camera parameters.
Parameter Optimization: Minimizes reprojection error for accurate calibration.
Save Calibration Data: Intrinsic parameters are saved in a CSV file for future use.

### Task 4: Compute Features for Each Major Region
Objective: Estimate the camera's position and orientation relative to the scene.
Implementation:
Position Estimation: Calculate rotation and translation matrices.
Visualize Changes: Monitor changes in the camera's position and orientation.

### Task 5: Project Outside Corners or 3D Axes
Objective: Project 3D axes onto the image frame based on the camera's estimated position.
Implementation:
3D to 2D Projection: Use projectPoints to project 3D points onto the 2D image plane.
Draw Axes Lines: Visualize 3D axes on the image frame.

### Task 6: Create a Virtual Object
Objective: Simulate the presence of 3D objects in the captured image frames.
Implementation:
Define Virtual Objects: Use 3D shapes like cylinders, pyramids, and cubes.
Project and Draw: Project 3D vertices onto the 2D plane and draw using OpenCV functions.

### Task 7: Detect Robust Features
Objective: Identify significant image features using the Harris corner detection method.
Implementation:
Corner Detection: Apply cornerHarris to compute the Harris corner response.
Normalization and Visualization: Normalize and visualize detected corners.

## Keypress Commands
'q': Quit the program.
's': Save the current calibration frame and perform calibration if frames >= 5.
'c': Save the current calibration data in a CSV file.
'x': Display 3D axes at the origin of world coordinates.
'd': Display 3D objects.
'h': Print the number of Harris Corners detected.
