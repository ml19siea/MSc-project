# import the necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import imutils
import argparse
import datetime
import numpy as np
from modules.detection import detect_people
from scipy.spatial import distance as dist
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import cv2
import math
from modules import distancing_config as config
import facenet
import detect_face
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf

# Construct the argument parser and parse the arguments for command line operations
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Load the COCO class labels on which YOLOv3 model was trained on
labelsPath ="./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Assign paths to the YOLOv3 weights and model configuration
weightsPath ="./yolo-coco/yolov3.weights"
configPath ="./yolo-coco/yolov3.cfg"

# Load YOLOv3 object detector trained on COCO dataset (80 classes)
print("Loading YOLOv3 weights from the disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Determine the *output* layer names that is needed from YOLOv3
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Initialize the video stream and pointer to output video file
print("Accessing input image...")
writer = None

fps_start = datetime.datetime.now()




#face recognition
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
frame = cv2.imread('./testing/104.jpg')
frame1=frame
writer = None
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7,0.8,0.8]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size =100 #1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')
        
        frame=imutils.resize(frame,width=700)
        print('Start Recognition')
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        faceNum = bounding_boxes.shape[0]
        print(faceNum)
        if faceNum > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]
            cropped = []
            scaled = []
            scaled_reshape = []
            for i in range(faceNum):
                emb_array = np.zeros((1, embedding_size))
                xmin = int(det[i][0])
                ymin = int(det[i][1])
                xmax = int(det[i][2])
                ymax = int(det[i][3])
                try:
                        # inner exception
                    if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                        print('Face is very close!')
                        continue
                    cropped.append(frame[ymin:ymax, xmin:xmax,:])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    print(model.predict_proba(emb_array))
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print("Predictions : [accuracy: {:.3f} ]".format(best_class_probabilities[0]))
                    if best_class_probabilities>0.6:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, result_names, (xmin,ymin+5), cv2.FONT_HERSHEY_COMPLEX_SMALL,.55, (0, 0, 0), thickness=1, lineType=1)
                                    
                    else :
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                        cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                        print("Predictions : [ name: {?} , accuracy: {:.3f} ]".format(best_class_probabilities[0]))

                except:   
                        
                    print("error")


results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

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
			if D[i, j] < 170:
					# update violation set with the indexes of the centroids
				violate.add(i)
				violate.add(j)


	# loop over the results
for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then initialize the color of the annotation
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