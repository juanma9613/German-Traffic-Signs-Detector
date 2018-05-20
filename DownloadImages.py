# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:54:13 2018

@author: LinaMaria
"""
import os
import urllib.request
import zipfile
import shutil
from sklearn.model_selection import train_test_split

def DownloadAndSplit(url='http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip',
                     zipname='FullDATA.zip',downloadZip=True,
                     unzipName='FullDATABASE',
                     folderData='FullIJCNN2013'
                     ):
    """
    It downloads the zip data and splits it into train and test data 
    INPUTS
    1.url to download the zip
    2.zipname name for the file
    3.downloadZip if you really want to download the zip 
    """
    new_wd=os.getcwd();
    os.chdir(new_wd)
    if(downloadZip):
        if (zipname in os.listdir(os.getcwd()+'/images' )):
            print('The zip has been downloaded previously')
        else:
            print('Downloading zip DATABASE')
            urllib.request.urlretrieve(url, 'images/'+zipname)    
    """
    EXTRACT DATA IF IT HAS NOT BEEN EXTRACTED
    """
    if (unzipName in os.listdir(os.getcwd()+'/images' )):
        print('the unzipping Process has been done previously')
    else:
        with zipfile.ZipFile('images/'+zipname) as Z :
            for elem in Z.namelist() :
                Z.extract(elem,'images/'+unzipName)
    directory_list = list()
    for root, dirs, files in os.walk(os.getcwd()+'/images'+'/'+unzipName+'/'+folderData , topdown=False):
        for name in dirs:
            directory_list.append(name)
    
    XFiles=list()
    Ylabels=list()
    if not os.path.exists(os.getcwd()+'/images/'+'tempFolder'):
            print('makingtempFolder')
            os.mkdir(os.getcwd()+'/images/'+'tempFolder')
    for fold in directory_list:
        files= os.listdir(os.getcwd()+'/images'+'/'+unzipName+'/'+folderData+'/'+fold)
        for name in files:
            XFiles.append(fold+name)
            Ylabels.append(fold)
            os.rename(os.getcwd()+'/images'+'/'+unzipName+'/'+folderData+'/'+fold+'/'+name,os.getcwd()+'/images/'+'tempFolder/'+fold+name)
    split(XFiles,Ylabels)
    with open(os.getcwd()+'/images'+'/'+unzipName+'/'+folderData+'/'+'ReadMe.txt') as f:
        data = f.readlines()
    data=data[-46:-3]
    f = open(os.getcwd()+'/images/Class.txt', 'w') # open for 'w'riting
    for i in data:
       f.write(i) # write text to file
    f.close()
    shutil.rmtree(os.getcwd()+'/images'+'/'+'tempFolder')
    shutil.rmtree(os.getcwd()+'/images'+'/'+unzipName)
    return
    
    
def split(XFiles,Ylabels,percentageTest=0.2):
    """
    A function that Split the data in two folders. 80% for trainning and 20% for testing
    it also renames XFiles  as Ylabels+Xfiles to avoid saving an object or txr with the labels
    INPUTS
    
    1.
    2.
    OUTPUTS
    *THE FILES IN THE TRAIN AND TEST FOLDER
    *A CLASS.TXT FILE WITH THE NUMBER AND NAME OF EACH CLASS
    """
    
    X_train, X_test, y_train, y_test =train_test_split(XFiles,Ylabels,test_size=percentageTest, random_state=64,stratify=Ylabels)
    for Xfile in X_train:
        try:
            os.rename(os.getcwd()+'/images/'+'tempFolder/'+Xfile,os.getcwd()+'/images'+'/train/'+Xfile)
        except:
            print('the file' +Xfile+ 'data is already in the train folder')
    for Xfile in X_test:
        try:
            os.rename(os.getcwd()+'/images/'+'tempFolder/'+Xfile,os.getcwd()+'/images'+'/test/'+Xfile)
        except:
           print('the file' +Xfile+ 'data is already in the test folder')


    return
