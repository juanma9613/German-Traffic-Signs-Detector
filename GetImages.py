# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:59:37 2018

@author: LinaMaria
"""

import os
#from testTF import testingTf
import numpy as np
from PIL import Image
import cv2 
#s = pickle.dumps(clf)
def getFlattenImagesAndLabels(DirectoryToSearch, TypeOfImage=1):
    """
   #################
   ### INPUTS ----------------1. DirectoryToSearch : A FOLDER IN ANY FOLDER OR SUBFOLDER OF THE
                              2. TypeOfImage 0 ----for test 1--- for train 2 for infer
   CURRENT WORKING DIRECTORY WHERE THERE ARE IMAGES.PPM
   
   #### OUTPUTS
   1 xImages:----------->numpy array size(nImages,32,32,1) 1 channel because images are grayscale
   2 Y_labels-----------> list with int class labels of len nImages
   ####A
   """
    new_wd=os.getcwd();
    os.chdir(new_wd)
    ## looking for DirectoryToSearch in path
    if os.path.exists(DirectoryToSearch) :
        files_Taken=os.listdir(DirectoryToSearch)
        files_Taken = [file for file in files_Taken if file.endswith("ppm")]
        trainFolder=DirectoryToSearch
    elif os.path.exists(os.getcwd()+DirectoryToSearch):
        trainFolder=os.getcwd()+DirectoryToSearch
        files_Taken =os.listdir(trainFolder)
        files_Taken = [file for file in files_Taken if file.endswith("ppm")]
    else:
        if(TypeOfImage==1):
            DirectoryToSearch='/images/train'
            print('\n INVALID DIRECTORY:\n taking files from images/train folder to avoid errors in training type "python app.py train --help" to avoid this error again\n\n\n')
        elif(TypeOfImage==0):
            DirectoryToSearch='/images/test'
            print('\n INVALID DIRECTORY:\n taking files from images/test folder to avoid errors in testing type "python app.py test --help" to avoid this error again\n\n\n')
        elif(TypeOfImage==2):
            DirectoryToSearch='/images/user'
            print('\n INVALID DIRECTORY:\n taking files from images/user folder to avoid errors in infering type "python app.py test --help" to avoid this error again\n\n\n')
            
        trainFolder=os.getcwd()+DirectoryToSearch
        files_Taken =os.listdir(trainFolder)
        files_Taken = [file for file in files_Taken if file.endswith("ppm")]
     ### creating a tensor for the Images and the labels in the model
    if len(files_Taken)==0:
        print('there arent any images to train the model')
    Y_labels=list()
    numbImages=len(files_Taken)
    ### creating a matrix
    xImages=np.zeros(shape=(numbImages,32*32))
    count=0
    ### RESIZING THE IMAGES TO 32.32 AND APPLYING GRAYSCALE FILTER
    for filename in files_Taken:
        im = Image.open(trainFolder+'/'+filename)
        final=preprocessImage(im)
        xImages[count]=final
        if(TypeOfImage==2):
            Y_labels.append(filename)
        else:
            Y_labels.append(int(filename[:2]))
        count+=1
    if(TypeOfImage==2):
        return xImages,Y_labels,trainFolder
    return xImages,Y_labels
    
    
def preprocessImage(im):
    """
    a function that converts an rgb image to a flattlen vector.
    Grayscale, reshape to 32*32, HisEqualization and uses np.flattlen()
    INPUTS 
    1.im --------> A Rgb image of any size
    
    OUTPUTS
    1.A GRAYSCALE FLATTLEN VECTOR SHAPE(32**2)
    
    """
    im = im.convert("L")
    im=np.array(im)
    preprocesedImage=cv2.resize(im,(32,32))
    preprocesedImage = cv2.equalizeHist(preprocesedImage)
    preprocesedImage=preprocesedImage.flatten()
#        preprocesedImage=np.divide(final, 255)
    return preprocesedImage
      
def getDictClasses():
    ClassDict=dict()
    with open(os.getcwd()+'/images/Class.txt') as f:
        data = f.readlines()
    data=[dat[:-1] for dat in data]
    for i in range(43):
        ClassDict[i]=data[i]
    return ClassDict