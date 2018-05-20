# -*- coding: utf-8 -*-
"""
Created on Sun May 13 14:02:39 2018

@author: LinaMaria
"""
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import GetImages

def LogRegressionTrain(DirectoryToTrain='/images/train',pathToSaveSc=os.getcwd()+'/models/model1/saved/scaler.pkl',
                       pathSaveLogRegModel=os.getcwd()+ '/models/model1/saved/modelScikit.pkl',Regularization=10):
    """
    A function that trains a LOGREG model with sklearn and saves the model as a pickle file
    ## it uses images in DirectoryToTrain by default but it's possible to change to any dir with a good amount of images
    INPUTS
    1.TrainImagesDirectory---> Directory to the images for training
    2.pathToSaveSc---------> the directory to save the standardSaler and the name of this with .pkl extension
    3.pathSaveLogRegModel-----> the directory to save the logRegModel and the name of this with .pkl extension
    4.Regularization---------> how stronger is the regularization to the weight in the function higher values
    means a a weak regularization
    """
    xImages,Y_labels=GetImages.getFlattenImagesAndLabels(DirectoryToTrain, TypeOfImage=1)
    sc = StandardScaler()
    sc.fit(xImages)
    X_train_std = sc.transform(xImages)
    ## C=100 taken because i want a weak regularization
    lr = LogisticRegression(C=Regularization, random_state=1)
    lr.fit(X_train_std, Y_labels)
    joblib.dump(sc,pathToSaveSc) 
    joblib.dump(lr,pathSaveLogRegModel) 
    print('\n Training accuracy:', lr.score(X_train_std, Y_labels))
    return
    
def LogRegressionTest(DirectoryToTest='/images/test',pathToLoadSc=os.getcwd()+'/models/model1/saved/scaler.pkl',
                       pathLoadLogRegModel=os.getcwd()+ '/models/model1/saved/modelScikit.pkl',Regularization=10):
    """
    A function that loads the skLOGREG model located in ModelPath 
    and uses it test the images in the Directory_Test
    INPUTS
    INPUTS
    1.Directory_infer---> Directory to the images that will be classified
    2.pathLoadLogRegModel---------> the directory to load the objeth
    3.pathToLoadSc-------------> the directory to load the standard scaler
    OUTPUT.
    IT PRINTS THE ACCURACY OF THE LOG REG MODEL IN THE TESTING IMAGES
    
    
    """
    xImages,Y_labels=GetImages.getFlattenImagesAndLabels(DirectoryToTest, TypeOfImage=0)
    SC= joblib.load(pathToLoadSc) 
    LR = joblib.load(pathLoadLogRegModel) 
    X_test_std = SC.transform(xImages)
    
    print('Test accuracy:', LR.score(X_test_std, Y_labels))
    return
def LogRegressionInfer(DirectoryToInfer='/images/user',pathToLoadSc=os.getcwd()+'/models/model1/saved/scaler.pkl',
                       pathLoadLogRegModel=os.getcwd()+ '/models/model1/saved/modelScikit.pkl',Regularization=10):
    """
    A function that loads the sklog regression model located in ModelPath 
    and uses it test the images in the Directory_infer folder 
    INPUTS
    1.Directory_infer---> Directory to the images that will be classified
    2.pathLoadLogRegModel---------> the directory to load the objeth
    3.pathToLoadSc-------------> the directory to load the standard scaler
    OUTPUT.
    IT OPENS A PLT PLOT WINDOW WITH THE ORIGINAL IMAGE AND THE OUTPUT CLASS
    """
    xImages,Y_labels,inferFolder=GetImages.getFlattenImagesAndLabels(DirectoryToInfer, TypeOfImage=2)
    SC= joblib.load(pathToLoadSc) 
    LR = joblib.load(pathLoadLogRegModel) 
    X_test_std = SC.transform(xImages)
    ypred = LR.predict(X_test_std)
    cont=0
    DictClasses=GetImages.getDictClasses()
    for yval in ypred:
            plt.figure(str(Y_labels[cont]))
            im = Image.open(inferFolder+'/'+Y_labels[cont])
            plt.text(-1,-1,'file'+Y_labels[cont]+'   belongs to class'+DictClasses[int(yval)])
            plt.imshow(im,vmin = 0, vmax = 255)
            cont=cont+1
    plt.show()
    return
    
    
    
