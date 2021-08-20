
# import the necessary packages
import os
import cv2
import imutils
import argparse
import datetime
import numpy as np
from modules import distancing_config as config
from modules.detection import detect_people
from scipy.spatial import distance as dist



# Construct the argument parser and parse the arguments for command line operations
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
args = vars(ap.parse_args())

labelsPath = "./yolov3/classes.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "./yolov3/yolov3_custom_final.weights"
configPath = "./yolov3/yolov3_test.cfg"

threshold=170

# Load YOLOv3 object detector trained on COCO dataset (80 classes)
print("Loading YOLOv3 weights from the disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# Check if we are going to use GPU
if config.USE_GPU:
	# Set CUDA as the preferable backend and target
	print("Setting-up preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Determine the *output* layer names that is needed from YOLOv3
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Initialize the video stream and pointer to output video file
print("Accessing video stream...")
vs = cv2.VideoCapture('./testing/new_clip_1.mp4')
writer = None


fps_start = datetime.datetime.now()
fps = 0
count_frames = 0

# loop over the video stream
while True:
	# read video stream
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, terminate the loop
	if not grabbed:
		break

	# resize frame and then detect people in the frame
	frame = imutils.resize(frame, width=700)
	count_frames = count_frames + 1
	results = detect_people(frame, net, ln, personIdx=LABELS.index("Actors"))

	# initialize the set of indexes that violate the minimum social distance
	violate = set()

	# ensure there are *at least* two people detections (required in order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the Euclidean distances between all pair of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

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
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if index pair exist in the violation set - update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	#Show social distancing violations on the output frame
	text = "Violations: {}".format(len(violate))

	fps_end = datetime.datetime.now()
	abs_time = fps_end - fps_start
	if abs_time.seconds == 0:
		fps = 0.0
	else:
		fps = (count_frames / abs_time.seconds)
	current_fps = "FPS: {:.2f}".format(fps)

	if (fps*480)<=0:
		lat = 0.0
	else:
		lat = np.around((1 / (fps*480)),5)
	latency = "Latency: {}".format(lat)

	if (len(results))==0:
		pred = 0
	else:
		pred = len(results)
	predict = "Total People: {}".format(pred)


	cv2.putText(frame, predict, (2, frame.shape[0] - 61), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
	cv2.putText(frame, current_fps, (2, frame.shape[0] - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
	cv2.putText(frame, latency, (2, frame.shape[0] - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
	cv2.putText(frame, text, (2, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



	# if output file path has been supplied and the video writer has not been initialized, do it now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

vs.release()
cv2.destroyAllWindows()
