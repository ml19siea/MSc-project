from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
import sys
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


import matplotlib.pyplot as plt



class training:
    def __init__(self, datadir, modeldir,classifier_filename):
        self.datadir = datadir
        self.modeldir = modeldir
        self.classifier_filename = classifier_filename

    def main_train(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                img_data = facenet.get_dataset(self.datadir)
                path, label = facenet.get_image_paths_and_labels(img_data)
                print('Classes: %d' % len(img_data))
                print('Images: %d' % len(path))

                facenet.load_model(self.modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                print('Extracting features of images for model')
                batch_size = 256
                image_size = 160 #160
                nrof_images = len(path)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = path[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_file_name = os.path.expanduser(self.classifier_filename)

                # Training Started
                print('Training Started')
                X=emb_array
                y=label
                #splitting the data
                x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=42)
                model = SVC(kernel='linear',probability=True)
                model=model.fit(x_train,y_train)
                print('training is completed')
                class_names = [cls.name.replace('_', ' ') for cls in img_data]
                #print(class_names)

                # Saving model
                with open(classifier_file_name, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('model is saved.')
                
                #testing the model
                print('testing classifer')
                with open(classifier_file_name,'rb') as infile:
                  (model,class_names)=pickle.load(infile)
                print('loaded classifier model file')
                predictions=model.predict_proba(x_test)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                
                #confusion matrix
                cm=confusion_matrix(y_test,best_class_indices)
                print(cm)

                accuracy=accuracy_score(y_test,best_class_indices)
                print('accuracy score of prediction is : ',accuracy)

                #precision score
                prec=precision_score(y_test,best_class_indices,average='weighted')
                print("precision is:",prec)

                #recall
                rec=recall_score(y_test,best_class_indices,average='weighted')
                print("recall is:",rec)

                #f1 score
                f1=f1_score(y_test,best_class_indices,average='weighted')
                print("f1 score is: ",f1)

                #plotting of confusion matrix
                print('Plotting of confusion matrix started')
                cm_plot=plot_confusion_matrix(model,x_test,y_test,labels=[0,1,2,3,4,5,6],
                display_labels=['Person1','Person2','Person3','Person4','Person5','Person6','Person7'],
                cmap=plt.cm.Reds)
                cm_plot.ax_.set_title('Confusion Matrix',color='red')
                plt.xlabel('Predicted label',color='red')
                plt.ylabel('True label',color='red')
                plt.gcf().axes[0].tick_params(colors='red')
                plt.gcf().axes[1].tick_params(colors='red')
                plt.gcf().set_size_inches(10,6)
                plt.savefig('confusion matrix.png')

                accuracy=accuracy_score(y_test,best_class_indices)
                print('accuracy of prediction is : ',accuracy)
                
                return(classifier_file_name)

