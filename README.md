# German-Traffic-Signs-Detector
Deep Learning Challenge

There are 3 models saved in this repo that were created to classify the data in The German Traffic Sign Detection Benchmark


The classification dataset has more than 1000 images of 43 classes


1. Clone this repository
2. make sure that you have the packages version given in requirements.txt
3. download and split the data running from the command line "python app.py download
4. Train the models or run inferences in the models saved in /model/model#/saved
To see how to train the models you can type python app.py --help
The easiest way to train a model is to type the model and the global path to the directory between double quotation marks ""
example:

python app.py train -m Model# -d "C:\Users\LinaMaria\Desktop\JUAN MANUEL\Machine learning\German Traffic Signs Detector\German-Traffic-Signs-Detector\images\train"
Model :
type Model1 or SKLogReg for the scikit logistic regression
     Model2 or TFLogReg for the tensorflow logistic regression
     Model3 or LeNet for the tensorflow Lenet CNN
5 to test a model use
python app.py test -m Model# -d "C:\Users\LinaMaria\Desktop\JUAN MANUEL\Machine learning\German Traffic Signs Detector\German-Traffic-Signs-Detector\images\test"

you can also test a model in images from another folder typing 
python app.py test -m Model# -d "C:\Global\Path\To\Test\Folder\test"
6 to infer with a model, use
python app.py infer -m Model# -d "C:\Users\LinaMaria\Desktop\JUAN MANUEL\Machine learning\German Traffic Signs Detector\German-Traffic-Signs-Detector\images\user"

you can also infer labels from images from another folder typing 
python app.py infer -m Model# -d "C:\Global\Path\To\Test\Folder\infer"

7.You can see more examples in the folder HowToRunFiles
8. To get more information about the models, read the reports in the folder reports.



