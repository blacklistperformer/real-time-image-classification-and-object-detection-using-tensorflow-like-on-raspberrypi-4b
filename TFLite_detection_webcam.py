# ------------------------------------------------------
# This code is written by Neeraj Nikhil Roy 20/11/EC/053 as a part of Project for BTech as a student of
# School of Enginnering, Jawaharlal Nehru University, New Delhi
# The guide for this project is Dr. Ankit Kumar Jaiswal
# This code is written for the purpose of Object Detection using Tensorflow Lite and OpenCV.
# The code is written for the purpose of detecting objects in real time using webcam.
# Title of the project is: REAL TIME IMAFE CLASSIFICATION AND OBJECT DETECTION USING TENSORFLOW LITE ON RASPBERRYPI 4B
# Members in my team are as follow
# Kshama Meena 20/11/EC/055 kshamameena7@gail.com
# Komal Kesav Nenavath 20/11/EC/012
# Divyansh Singh 20/11/EC/057
# Github link to the repository https://github.com/blacklistperformer/real-time-image-classification-and-object-detection-using-tensorflow-like-on-raspberrypi-4b
# Link to my other socials
# Instagram: https://www.instagram.com/blacklistperformer/
# Linkedln: https://www.linkedin.com/in/neeraj-roy-556968192/
# Github: https://github.com/BlackListPerformer
# Stackoverflow: https://stackoverflow.com/users/19916561/neeraj-roy
# Email: neerajroy06502@gmail.com

# Kshama Meena's social
# Linkedln: https://www.linkedin.com/in/kshama-meena-1851a8207/
# Github: https://github.com/kshamameena

# Komal's Social: Github: https://github.com/komalkesav

# ------------------------------------------------------


# IMPORTING LIABRARIES
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


#------------------------------------------------------
# DEFINE VIDEOTREAM CLASS
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        # Setting camera parameters like resolution and framerate
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
    # Initializing variables for camera control
	# Variable to control when the camera is stopped
        self.stopped = False

    #This class is designed to handle video streaming from a webcam in a separate processing thread. It uses OpenCV
    # (cv2) to capture video frames.

#------------------------------------------------------

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self
    #This method starts a new thread to continuously read frames from the video stream.

    #This method is called from the constructor, so the camera starts automatically when the VideoStream object is
    # created. This method runs in a loop, continuously updating the frame in the background.
    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    #This method returns the most recent frame captured by the camera.
    def read(self):
	# Return the most recent frame
        return self.frame

    #This method stops the thread and releases camera resources.
    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

#------------------------------------------------------
# ARGUMENT PARSING
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()
#These lines set up an argument parser to take inputs from the command line. Users can specify the model directory,
# graph file, label map file, confidence threshold, webcam resolution, and whether to use the Edge TPU for acceleration.


#------------------------------------------------------
# MODEL AND LABEL LOADING
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

#These lines retrieve the values of the command-line arguments and set up variables for the model, graph, label map,
#confidence threshold, webcam resolution, and Edge TPU usage.


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# These lines check if the tflite_runtime library is available. If yes, it imports the Interpreter class, and if Edge
# TPU is used, it imports the load_delegate function. If tflite_runtime is not available, it falls back to importing
# from the regular TensorFlow library.


# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'
# If Edge TPU is used, and the graph file is set to the default 'detect.tflite', it changes the graph file to
# 'edgetpu.tflite'.

#------------------------------------------------------
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
# These lines construct the full paths to the model and label map files based on the current working directory and the
#  provided arguments.

#------------------------------------------------------
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
# These lines load the label map, which maps each class name to an integer id. The integer ids correspond to the


# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])
#If the first label is '???', it is removed. This is a fix for label maps from the COCO "starter model."

#------------------------------------------------------
# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

# It initializes the TensorFlow Lite interpreter, taking into account the use of Edge TPU if specified.

#Allocates tensors for the model.
interpreter.allocate_tensors()

#------------------------------------------------------
# MODEL DETAILS
# Get input and output tensors
# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
# These lines retrieve details about the input and output tensors of the model, as well as the height and width of
# the input


#It checks if the model is a floating-point model and sets the input mean and standard deviation accordingly.
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2
# These lines determine the indices for bounding boxes, classes, and scores based on the output layer name. This is done
#  to handle differences between TF1 and TF2 model outputs.

#------------------------------------------------------
# VIDEO STREAM INITIALIZATION
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#------------------------------------------------------
# MAIN LOOP
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Timing the Loop
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # This line records the start time of the loop using OpenCV's getTickCount() function. This timing information will
    #  be used later to calculate the frame processing time and, consequently, the frames per second (FPS) of the video
    #  stream.


    # Grab frame from video stream
    frame1 = videostream.read()
    # This line reads the most recent frame from the videostream object, which represents the video stream from the
    #  webcam.



    # Acquiring Video Frame
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    #  It makes a copy of the frame (frame1.copy()) to avoid modifying the original frame. Then, it converts the frame from
    #  BGR to RGB color space, resizes it to match the expected input size of the model, and adds an extra dimension to
    #  represent the batch size (np.expand_dims). The resulting input_data is the preprocessed frame ready to be fed
    #  into the TensorFlow Lite model for inference.


    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std


    # Performing Object Detection
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # These lines set the input tensor of the TensorFlow Lite model to the preprocessed frame (input_data) and invoke
    # the interpreter to perform inference.


    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    # These lines retrieve the model outputs from the output tensors. The output tensors are indexed using the indices
    # The boxes, classes, and scores represent the bounding box coordinates, class indices, and confidence scores of the
    # detected objects, respectively.


    # Loop over all detections and draw detection box if confidence is above minimum threshold
    # This loop iterates through the detected objects, checks if the confidence score is above a specified threshold,
    # and if so, it draws bounding boxes around the objects and adds labels with the object's class and confidence
    # score.
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within
            # image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


    # Calculating and Displaying Frame Rate
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculating and Displaying Frame Rate
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

#------------------------------------------------------
# CLEANUP
cv2.destroyAllWindows()
videostream.stop()
# These lines clean up by closing all OpenCV windows and stopping the video stream.
#------------------------------------------------------
