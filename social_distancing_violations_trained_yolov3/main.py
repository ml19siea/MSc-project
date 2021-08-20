# python main.py --input newvid.mp4 --output output.avi

# import the necessary packages
import os
import cv2
import imutils
import argparse
import datetime
import numpy as np
from modules.detection import detect_people
from scipy.spatial import distance as dist

###############################################################################################
# Construct the argument parser and parse the arguments for command line operations
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())


threshold=170

# Load the COCO class labels on which YOLOv3 model was trained on
labelsPath = "./yolov3/classes.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Assign paths to the YOLOv3 weights and model configuration
weightsPath = "./yolov3/yolov3_custom_final.weights"
configPath = "./yolov3/yolov3_test.cfg"

# Load YOLOv3 object detector trained on COCO dataset (80 classes)
print("Loading YOLOv3 weights from the disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# Determine the *output* layer names that is needed from YOLOv3
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Initialize the video stream and pointer to output video file
print("Accessing an image...")
frame=cv2.imread("./testing/470.jpg")
writer = None

frame = imutils.resize(frame, width=700)
results = detect_people(frame, net, ln, personIdx=LABELS.index("Actors"))

	# initialize the set of indexes that violate the minimum social distance
violate = set()

	# ensure there are *at least* two people detections (required in order to compute our pairwise distance maps)
if len(results) >= 2:
		# extract all centroids from the results and compute the Euclidean distances between all pair of the centroids
	centroids = np.array([r[2] for r in results])
	D = dist.cdist(centroids, centroids, metric="euclidean")
	print(D)

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
	print(prob)
	(startX, startY, endX, endY) = bbox
	(cX, cY) = centroid
	color = (0, 255, 0)
	text="Safe"

		# if index pair exist in the violation set - update the color
	if i in violate:
		color = (0, 0, 255)
		text="Alert"

		# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
	cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	cv2.putText(frame, text, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		#cv2.circle(frame, (cX, cY), 5, color, 1)

	#Show social distancing violations on the output frame
text = "Violations: {}".format(len(violate))

if (len(results))==0:
	pred = 0
else:
	pred = len(results)
predict = "Total People: {}".format(pred)


cv2.putText(frame, predict, (2, frame.shape[0] - 61), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
cv2.putText(frame, text, (2, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.imwrite('output.jpg',frame)
cv2.waitKey()