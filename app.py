import click

#from download import downloadDATA
import testAndInferLeNet 
#from testTF import testingTf
import LeNet5
import LogRegTensorFlow
import SKLogRegression
import DownloadImages
@click.group()
def main():
    pass

@main.command("download")
def download():
    DownloadImages.DownloadAndSplit()
   
@main.command("train")
@click.option('-m', '--model',help=' Enter Model1 or SKLogReg for the scikit logistic regresion \
              \n\n or Enter Model2 or TFLogReg for the Tensorflow logistic regresion \
              \n\n or Enter Model3 or LeNet for the Tensorflow LeNetImplementation')
@click.option('-d', '--directory',help='enter a global path between "" symbol and use \
                          \ between folders example "C:\Who\Desktop\GermanSigns\images\\train" \n\n\n or You can also give a relative path \n inside the Working directory example: "/images/train"')
def train(model, directory):
    if (model=='Model1' or model=='SKLogReg'):
        SKLogRegression.LogRegressionTrain(DirectoryToTrain=directory)
    elif (model=='Model2' or model=='TFLogReg'):
        LogRegTensorFlow.LogRegTrain(TrainImagesDirectory=directory)
    elif (model=='Model3'or model=='LeNet'):
        LeNet5.TrainAndSave(Directory_Train=directory,EPOCHS=30)
@main.command("test")
@click.option('-m', '--model',help=' Enter Model1 or SKLogReg for the scikit logistic regresion \
              \n\n or Enter Model2 or TFLogReg for the Tensorflow logistic regresion \
              \n\n or Enter Model3 or LeNet for the Tensorflow LeNetImplementation')
@click.option('-d', '--directory',help='enter a global path between "" symbol and use \
                          \ between folders example "C:\Who\Desktop\GermanSigns\images\\test" \n\n\n or You can also give a relative path \n inside the Working directory example: "/images/test"')
def test(model, directory):
    if (model=='Model1' or model=='SKLogReg'):
        SKLogRegression.LogRegressionTest(DirectoryToTest=directory)
    elif (model=='Model2' or model=='TFLogReg'):
        LogRegTensorFlow.testLogReg(Directory_Test=directory)    
    elif (model=='Model3'or model=='LeNet'):
        testAndInferLeNet.testLeNet5(Directory_Test='/images/test' ,ModelPath='/models/model3/saved/')
#    msg = "I should be testing model {} with data from directory {}"
@main.command("infer")
@click.option('-m', '--model',help=' Enter Model1 or SKLogReg for the scikit logistic regresion \
              \n\n or Enter Model2 or TFLogReg for the Tensorflow logistic regresion \
              \n\n or Enter Model3 or LeNet for the Tensorflow LeNetImplementation')
@click.option('-d', '--directory',help='enter a global path between "" symbol and use \
                          \ between folders example "C:\Who\Desktop\GermanSigns\images\\user" \n\n\n or You can also give a relative path \n inside the Working directory example: "/images/user"')
def infer(model, directory):
    if (model=='Model1' or model=='SKLogReg'):
            SKLogRegression.LogRegressionInfer(DirectoryToInfer=directory)
    elif (model=='Model2' or model=='TFLogReg'):
            LogRegTensorFlow.inferLogReg(Directory_infer=directory)    
    elif (model=='Model3'or model=='LeNet'):
            testAndInferLeNet.inferLeNet5(Directory_infer=directory)
    

    
if __name__ == '__main__':
    main(obj={})