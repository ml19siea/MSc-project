import argparse

import cv2
import numpy as np
import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
args = vars(ap.parse_args())

weightsPath = "./yolov3_custom_final.weights"
configPath = "./yolov3_custom_test.cfg"
labels='./classes.names'


CONF_THRESH, NMS_THRESH = 0.8, 0.5

# Load the network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Initialize the video stream and pointer to output video file
print("Accessing video stream...")
vs = cv2.VideoCapture('./new_clip_1.mp4')
writer = None

fps_start = datetime.datetime.now()
fps = 0
count_frames = 0

#loop over video stream
while True:
    grabbed, img = vs.read()
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    class_ids, confidences, b_boxes = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]


            if confidence > CONF_THRESH:   
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                #print(confidence)
                class_ids.append(int(class_id))
                #print(class_ids)

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
    if len(indices)>0:
        indices=indices.flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        #print(confidences[index])
        #print(class_ids[index])
        cv2.rectangle(img, (x, y), (x + w, y + h),(0,255,255), 2)
        cv2.putText(img, classes[class_ids[index]], (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, .70,(0,0,0), 2)


    # if output file path has been supplied and the video writer has not been initialized, do it nowf
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,(img.shape[1], img.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(img)

vs.release()
cv2.destroyAllWindows()