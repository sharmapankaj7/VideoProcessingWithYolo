import cv2
import numpy as np

# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

# Define the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Define the classes
classes = ["phone"]

# Open the video
cap = cv2.VideoCapture('test_short.mp4')

# Initialize variables to keep track of phone usage
phone_in_hand = False
frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Get the dimensions of the frame
    height, width, channels = frame.shape

    # Convert the image to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the blob as the input to the network
    net.setInput(blob)

    # Get the output from the network
    outs = net.forward(output_layers)

    # Initialize the list of detected objects
    objects = []

    # Loop over the output from the network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                # Get the bounding box for the object
                x = int(detection[0] * width)
                y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Add the object to the list of detected objects
                objects.append((x, y, w, h))

    # Draw a rectangle around the objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Check if a phone is in the frame
    if len(objects) > 0:
        phone_in_hand = True
        frame_count += 1
    else:
        phone_in_hand = False
        frame_count = 0

    # Check if the phone was in the frame for a certain number of frames
    if phone_in_hand:
        if frame_count > 30: # Number of frames to consider phone in use
            print("Phone in use for more than 30 frames")
    else:
        print("Phone not in use")
