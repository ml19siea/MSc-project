import argparse
import cv2
import numpy as np
import os
import cv2
import imutils
import argparse
import datetime
from modules.detection import detect_people
from scipy.spatial import distance as dist

#face recognition 

# Construct the argument parser and parse the arguments for command line operations
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
args = vars(ap.parse_args())

face_config='./yolo_face/yolov3_custom_test.cfg'
face_weight='./yolo_face/yolov3_custom_final.weights'
face_names='./yolo_face/classes.names'

CONF_THRESH, NMS_THRESH = 0.8, 0.5

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
vs = cv2.VideoCapture("./testing/new_clip_1.mp4")
writer = None

fps_start = datetime.datetime.now()
fps = 0
count_frames = 0

while True:
	grabbed,img=vs.read()
	img = imutils.resize(img, width=700)
	height, width = img.shape[:2]

	# Load the network
	net= cv2.dnn.readNetFromDarknet(face_config, face_weight)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	# Get the output layer from YOLO
	layers = net.getLayerNames()
	output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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
	indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()
	#print(indices)

	# Draw the filtered bounding boxes with their class to the image
	with open(face_names, "r") as f:
	    classes = [line.strip() for line in f.readlines()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	for index in indices:
	    x, y, w, h = b_boxes[index]
	    #print(confidences[index])
	    #print(class_ids[index])
	    cv2.rectangle(img, (x, y), (x + w, y + h),(0,255,255), 2)
	    cv2.putText(img, classes[class_ids[index]], (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, .70,(0,0,0), 2)

	#social distancing violations

	threshold=170

	# Load the COCO class labels on which YOLOv3 model was trained on
	labelsPath = "./yolo_obj/classes.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# Assign paths to the YOLOv3 weights and model configuration
	weightsPath = "./yolo_obj/yolov3_custom_final.weights"
	configPath = "./yolo_obj/yolov3_test.cfg"

	# Load YOLOv3 object detector trained on COCO dataset (80 classes)
	#print("Loading YOLOv3 weights from the disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


	# Determine the *output* layer names that is needed from YOLOv3
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	results = detect_people(img, net, ln, personIdx=LABELS.index("Actors"))

		# initialize the set of indexes that violate the minimum social distance
	violate = set()

		# ensure there are *at least* two people detections (required in order to compute our pairwise distance maps)
	if len(results) >= 2:
			# extract all centroids from the results and compute the Euclidean distances between all pair of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		#print(D)

			# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
					# check if the distance between any two centroid pairs is less than the configured number of pixels
				if D[i, j] < threshold:
						# update violation set with the indexes of the centroids
					violate.add(i)
					violate.add(j)


		# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then initialize the color of the annotation
		#print(prob)
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		text="Safe"

			# if index pair exist in the violation set - update the color
		if i in violate:
			color = (0, 0, 255)
			text="Alert"

			# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
		cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
		cv2.putText(img, text, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

		#Show social distancing violations on the output frame
	text = "Violations: {}".format(len(violate))

	if (len(results))==0:
		pred = 0
	else:
		pred = len(results)
	predict = "Total People: {}".format(pred)


	cv2.putText(img, predict, (2, img.shape[0] - 61), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
	cv2.putText(img, text, (2, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,(img.shape[1], img.shape[0]), True)
	if writer is not None:
		writer.write(img)

vs.release()
cv2.destroyAllWindows()
