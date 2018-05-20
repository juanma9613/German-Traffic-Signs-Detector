# -*- coding: utf-8 -*-
"""
Created on Sun May 13 14:02:39 2018

@author: LinaMaria
"""

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
import os
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.utils import shuffle
import GetImages
from sklearn.model_selection import train_test_split

def SCReg(DirectoryToTrain):
    xImages,Y_labels=GetImages.getFlattenImagesAndLabels(DirectoryToSearch='/images/train', TypeOfImage=1)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    ## C=100 taken because i want a weak regularization
    lr = LogisticRegression(C=10, random_state=1)
    lr.fit(X_train_std, Y_train)
    joblib.dump(sc,'/models/model1/saved/scaler.pkl') 
    joblib.dump(lr, '/models/model1/saved/modelScikit.pkl') 
    print('Training accuracy:', lr.score(X_train_std, Y_train))
    
    return
def SCEval(DirecToTest):
    new_wd=os.getcwd();
    os.chdir(new_wd)
    for root, dirs, files in os.walk(os.getcwd()):
        if DirecToTest in dirs:
            testFolder=os.path.join(root, DirecToTest)
            print('the test folder is'+testFolder)
    files_Taken =os.listdir(testFolder)
    files_Taken = [file for file in files_Taken if file.endswith("ppm")]
    #    print((files))
    ##removing only .keep
#    files.remove('.keep')   
    #taking only .ppm endswith
    #for file in files
    X_test=list()
    Y_test=list()
    for filename in files_Taken:
        im = Image.open(testFolder+'/'+filename)
        im = im.convert("L")
        im=np.array(im)
        imScaled=cv2.resize(im,(32,32))
        final=imScaled.flatten()
        X_test.append(final)
        Y_test.append(filename[:2])
    print(len(X_test))
    print(len(Y_test))
    SC= joblib.load('scaler.pkl') 
    LR = joblib.load('modelScikit.pkl') 
    X_test_std = SC.transform(X_test)
    print('Test accuracy:', LR.score(X_test_std, Y_test))
    return
    

    



#    elif len(str(i))==2:
#        filename='000'+str(i)+'.ppm'
#    im = Image.open(filename)
#    im.show()
#ROIS=pd.read_table('gt.txt', delimiter=';',header=None)
##ROIS[6] = [np.zeros(32**2) for x in range(len(ROIS))]
#Y_DATA=np.array(ROIS.loc[:,5])
#X_DATA=np.zeros([len(ROIS),32**2])
#for i in range(len(ROIS)):
#    filename,xmin,ymin,xmax,ymax=ROIS.loc[i,0:4]
#    im = Image.open(filename)
#    im = im.convert("L")
##    im.save('bw.png')
#    im=np.array(im)
#    ## así coge el primer pixel pero no el ultimo
#    imroi=im[ymin:ymax,xmin:xmax]
#    imroi=cv2.resize(imroi,(32,32))
#    # IMROI RESHAPE
#    # cv2.resize
#    ## IMROI FLATTEN
#    final=imroi.flatten()
#    X_DATA[i]=final
##    plt.figure(i+1)
##    print(ymax-ymin,xmax-xmin)
##    implot=plt.imshow(imroi)
##    plt.imshow(imroi,cmap="hot")
##    im.save('bw.png')
#    ## hay que pasarlas a blanco y negro.
#    ## aplanarlas en vectores luego
#    ## crear XTRAIN
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size=0.2, random_state=56,stratify=Y_DATA)
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#sc.fit(X_train)
#X_train_std = sc.transform(X_train)
#X_test_std = sc.transform(X_test)
#from sklearn.linear_model import LogisticRegression
## numb for random weight ini
#lr = LogisticRegression(C=100.0, random_state=1)
#lr.fit(X_train_std, y_train)
#
###
##EVALUACION DETECTAR EL TIPO DE SEÑAL
#lr.predict_proba(X_test_std[:, :]).argmax(axis=1)
#print('Training accuracy:', lr.score(X_train_std, y_train))
### detectar el score
#print('Test accuracy:', lr.score(X_test_std, y_test))
#
#    
#import os
#cwd = os.getcwd()

#import urllib.request
#import zipfile
##url='http://www.mynikko.com/dummy/dummy12.zip'
#url='http://file-examples.com/wp-content/uploads/2017/02/zip_5MB.zip'
#
## Download the file from `url` and save it locally under `file_name`:
#urllib.request.urlretrieve(url, 'file5mb.zip')
### path to file
#zipname='file5mb.zip'
#zipname='FullIJCNN2013.zip'
#with zipfile.ZipFile(zipname) as Z :
#    for elem in Z.namelist() :
#        ## ad path
#        Z.extract(elem,'unzippedDATABASE')
     #   print(elem)

    
    
