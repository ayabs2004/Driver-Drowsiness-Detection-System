# Driver-Drowsiness-Detection-System
A desktop application that analyzes photos of drivers to detect signs of drowsiness and fatigue in real time.
How it works
The system uses a three-stage pipeline:

1-Face detection — OpenCV's DNN face detector (with dlib HOG as fallback) locates the driver's face in the image
2-Facial landmark detection — dlib's 68-point landmark predictor precisely locates the eyes and mouth on the detected face, replacing unreliable HAAR cascade eye detection
3-CNN classification — a custom-trained Convolutional Neural Network classifies each eye as open or closed and the mouth as open (yawning) or closed


