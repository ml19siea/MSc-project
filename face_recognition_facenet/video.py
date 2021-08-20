
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import math
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

#Initialize the video stream and pointer to output video file
print("Accessing video stream...")
vs = cv2.VideoCapture("./testing/new_clip_1.mp4")
writer = None

fps_start = datetime.datetime.now()
fps = 0
count_frames = 0


while True:
    # read video stream
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, terminate the loop
    if not grabbed:
        break

    # resize frame and then detect people in the frame
    frame = imutils.resize(frame, width=700)
    count_frames = count_frames + 1

 #face recognition
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'
    npy='./npy'
    train_img="./train_img"
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
            print('Start Face Recognition')
            if frame.ndim==2:
                frame=facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]

            if faceNum==0:
                print("No face is recognised")

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
                        cropped.append(frame[ymin:ymax, xmin:xmax,:])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print("Predictions : [accuracy: {:.3f} ]".format(best_class_probabilities[0]))

                        if best_class_probabilities>0.6:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255, 0), 2)    #boxing face
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                    cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0,255,255), -1)
                                    cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                             .55, (0, 0, 0), thickness=1, lineType=1)
                            else:
                                #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                                #cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0,255,255), -1)
                                print("Predictions : [ name: {?} , accuracy: {:.3f} ]".format(best_class_probabilities[0]))
                    except:
                        print('error')

    fps_end = datetime.datetime.now()
    abs_time = fps_end - fps_start
    if abs_time.seconds == 0:
        fps = 0.0
    else:
        fps = (count_frames / abs_time.seconds)
    current_fps = "FPS: {:.2f}".format(fps)

    faceNum= "Total faces recognised : {}".format(faceNum)

    cv2.putText(frame, current_fps, (2, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
    cv2.putText(frame, faceNum, (2, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 2)

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





