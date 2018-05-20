# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:23:35 2018

@author: LinaMaria
"""
#from download import downloadDATA
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from LeNet5 import getImagesAndLabels
import LeNet5
import GetImages
def evaluate(X,Y,acOp,X_data, y_data):
    """A FUNCTION TO EVALUATE THE ACCURACY OF THE CNN IN THE TRAINNING DATA
    INPUTS 
    1. x tfplace holder size(?,32,32,1)
    2. y tfplace holder size(?)
    3.accuracy_operation ---->the operation in the NN to determine wheter a N fails 
    4.X_data np.array(Nimages,32,32,1)
    5 y_data list of len (Nimages) 
    6.BATCH_SIZE size of the batch to evaluate the accuracy
    
    OUTPUT 
    1 total_accuracy  OF THE NN
    """
    sess = tf.get_default_session()
    accuracy=sess.run(acOp, feed_dict={X: X_data, Y: y_data})
    return accuracy

def testLeNet5(Directory_Test='/images/test' ,ModelPath='/models/model3/saved/'):
    """
    A function that loads the lenet5 model located in ModelPath 
    and uses it test the images in the Directory_Test
    INPUTS
    1.Directory_Test---> Directory to the images for testing
    2.ModelPath---------> the directory to load the tf graph
    OUTPUT.
    IT PRINTS THE ACCURACY OF THE MODEL IN THE TESTING IMAGES
    
    
    """
    TypeOfImage=0
    X_test,Y_test=LeNet5.getImagesAndLabels(Directory_Test,TypeOfImage)
    with tf.Session() as sess:   
        loader = tf.train.import_meta_graph(os.getcwd()+ModelPath+'.meta')
        loader.restore(sess, tf.train.latest_checkpoint(os.getcwd()+ModelPath))
        graph = tf.get_default_graph()
        X=graph.get_tensor_by_name("X:0")
        Y=graph.get_tensor_by_name("Y:0")
        acOp=graph.get_tensor_by_name("accuracyOp:0")
        validation_accuracy=evaluate(X,Y,acOp,X_test, Y_test)
        print("\n\nTest Accuracy = {:.2f}%".format( (validation_accuracy*100)))
    return
def inferLeNet5(Directory_infer='/images/user' ,ModelPath='/models/model3/saved/'):
    """
    A function that loads the lenet5 model located in ModelPath 
    and uses it test the images in the Directory_infer folder 
    INPUTS
    1.Directory_infer---> Directory to the images that will be classified
    2.ModelPath---------> the directory to load the tf graph
    OUTPUT.
    IT OPENS A PLT PLOT WINDOW WITH THE ORIGINAL IMAGE AND THE OUTPUT CLASS
    """
    TypeOfImage=2
    X_test,Y_test,inferFolder=LeNet5.getImagesAndLabels(Directory_infer,TypeOfImage)
    with tf.Session() as sess:   
        loader = tf.train.import_meta_graph(os.getcwd()+ModelPath+'.meta')
        loader.restore(sess, tf.train.latest_checkpoint(os.getcwd()+ModelPath))
        graph = tf.get_default_graph()
        X=graph.get_tensor_by_name("X:0")
        Logits=graph.get_tensor_by_name("logits:0")
        cont=0
        DictClasses=GetImages.getDictClasses()
        for i in X_test:
            i=i.reshape(1,32,32,1)
            Proba=sess.run(Logits, feed_dict={X: i})
            plt.figure(str(Y_test[cont])+'   '+str(np.argmax(Proba,1)))
            im = Image.open(inferFolder+'/'+Y_test[cont])
            plt.text(-1,-1,'file'+Y_test[cont]+'   belongs to class '+DictClasses[int(np.argmax(Proba,1))])
            plt.imshow(im,vmin = 0, vmax = 255)
#            plt.imshow(i[0,:,:,0],cmap='gray', vmin = 0, vmax = 1)
#                plt.show()
            cont=cont+1
    plt.show()
    return
    


