# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:59:37 2018

@author: LinaMaria
"""
#from download import downloadDATA
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
#from download import downloadDATA
import matplotlib.pyplot as plt
#from LeNet5 import getImagesAndLabels
import GetImages



def evaluate(xTensor,yTensor,accuracy_operation,X_data, y_data, BATCH_SIZE=64):
    """A FUNCTION TO EVALUATE THE ACCURACY OF THE LOGREG IN THE DATA
    INPUTS 
    1. x tfplace holder size(?,32*32)
    2. y tfplace holder size(?)
    3.accuracy_operation ---->the operation in the Logreg to determine if a LOGREG fails 
    4.X_data np.array(Nimages,32*32)
    5 y_data list of len (Nimages) 
    6.BATCH_SIZE size of the batch to evaluate the accuracy
    
    OUTPUT 
    1 total_accuracy / num_examples IT IT THE ACCURACY OF THE LOGREG
    OUTPUT
    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    #    print(X_data.shape)
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy=sess.run(accuracy_operation, feed_dict={xTensor: batch_x, yTensor: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#s = pickle.dumps(clf)
def LogRegTrain(TrainImagesDirectory='/images/train',pathToSave=os.getcwd()+'/models/model2/saved/', BATCH_SIZE=64,EPOCHS=50,learning_rate=0.01,
        n_out=43):       
    """
    A function that trains a LOGREG model with tf and saves the model after the final epoch
    ## it uses images in TrainImagesDirectory by default but it's possible to change to any dir with a good amount of images
    INPUTS
    1.TrainImagesDirectory---> Directory to the images for training
    2.pathToSave---------> the directory to save the tf graph
    3.BATCH_SIZE-----> batch size for the stochastic gradient APPROACH, 
    4.EPOCHS---------> number of iterations of the training
    5.learning_rate----> learning rate for the optimizer
    
    """
    xImages,Y_labels=GetImages.getFlattenImagesAndLabels(TrainImagesDirectory, TypeOfImage=1)
    print('training TF LogReg with '+str(len(Y_labels))+' input files')
    g = tf.Graph()
    with g.as_default():
        xTensor = tf.placeholder(tf.float32, (None, 32*32),name='X')
        yTensor = tf.placeholder(tf.int32, (None),name="Y")
        weight = tf.Variable(tf.truncated_normal(shape=[32*32, n_out], stddev=0.01), name="weights")
        bias = tf.Variable(tf.zeros([1, n_out]), name="bias")
        logits = tf.add(tf.matmul(xTensor, weight),bias,name='logits')
        one_hot_y = tf.one_hot(yTensor, n_out)
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss = tf.reduce_mean(entropy) # computes the mean over samples in the batch
        optimizer =tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1),name='correct')
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracyOp')
        saver = tf.train.Saver()
    with tf.Session(graph=g) as sess:
        sess.run(init)
        num_samples = len(Y_labels)
        print("Training your Log Reg with Tensorflow ...\n")
        for i in range(EPOCHS):
            shuffled_images, y_train = shuffle(xImages, Y_labels)
            for offset in range(0, num_samples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = shuffled_images[offset:end], y_train[offset:end]
                sess.run(optimizer, feed_dict={xTensor: batch_x, yTensor: batch_y})
            validation_accuracy = evaluate(xTensor,yTensor,accuracy_operation,xImages, Y_labels)
    #IN CASE VALIDATION ON TEST DATA         validation_accuracy2 = evaluate(X_test, ytest)
#            print("training RESULT AT EPOCH # {} : Validation Accuracy = {:.2f}%".format(i+1, (validation_accuracy*100)))
        print("\n\n FINAL RESULT AT EPOCH # {} : Validation Accuracy = {:.2f}%".format(i+1, (validation_accuracy*100)))
    #IN CASE VALIDATION ON TEST DATA        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy2*100)))
        saver.save(sess,pathToSave)
    return
    
def testLogReg(Directory_Test='/images/test' ,ModelPath='/models/model2/saved/'):
    """
    A function that loads the LOGREG model located in ModelPath 
    and uses it test the images in the Directory_Test
    INPUTS
    1.Directory_Test---> Directory to the images for testing
    2.ModelPath---------> the directory to load the tf graph
    OUTPUT.
    IT PRINTS THE ACCURACY OF THE LOG REG MODEL IN THE TESTING IMAGES
    
    
    """
    TypeOfImage=0
    X_test,Y_test=GetImages.getFlattenImagesAndLabels(Directory_Test,TypeOfImage)
    print('testing TF LogReg with '+str(len(Y_test))+' input files')
    with tf.Session() as sess:   
        loader = tf.train.import_meta_graph(os.getcwd()+ModelPath+'.meta')
        loader.restore(sess, tf.train.latest_checkpoint(os.getcwd()+ModelPath))
        graph = tf.get_default_graph()
        X=graph.get_tensor_by_name("X:0")
        Y=graph.get_tensor_by_name("Y:0")
        acOp=graph.get_tensor_by_name("accuracyOp:0")
        validation_accuracy=evaluate(X,Y,acOp,X_test, Y_test, BATCH_SIZE=64)
        print("\n\nTest Accuracy = {:.2f}%".format( (validation_accuracy*100)))
    return
def inferLogReg(Directory_infer='/images/user' ,ModelPath='/models/model2/saved/'):
    """
    A function that loads the LOGREG model located in ModelPath 
    and uses it test the images in the Directory_infer folder 
    INPUTS
    1.Directory_infer---> Directory to the images that will be classified
    2.ModelPath---------> the directory to load the tf graph
    OUTPUT.
    IT OPENS A PLT PLOT WINDOW WITH THE ORIGINAL IMAGE AND THE OUTPUT CLASS
    """
    TypeOfImage=2
    X_test,Y_test,inferFolder=GetImages.getFlattenImagesAndLabels(Directory_infer,TypeOfImage)
    print(X_test.shape)
    with tf.Session() as sess:   
        loader = tf.train.import_meta_graph(os.getcwd()+ModelPath+'.meta')
        loader.restore(sess, tf.train.latest_checkpoint(os.getcwd()+ModelPath))
        graph = tf.get_default_graph()
        X=graph.get_tensor_by_name("X:0")
        Logits=graph.get_tensor_by_name("logits:0")
        cont=0
        DictClasses=GetImages.getDictClasses()
        for i in X_test:
            i=i.reshape(1,32**2)
            Proba=sess.run(Logits, feed_dict={X: i})
            plt.figure(str(Y_test[cont]))
            im = Image.open(inferFolder+'/'+Y_test[cont])
            plt.text(-1,-1,'file'+Y_test[cont]+'   belongs to class'+DictClasses[int(np.argmax(Proba,1))])
            plt.imshow(im,vmin = 0, vmax = 255)
#            plt.imshow(i[0,:,:,0],cmap='gray', vmin = 0, vmax = 1)
#                plt.show()
            cont=cont+1
    plt.show()
    return


    
    
    
