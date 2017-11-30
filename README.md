# Road-sign-Arrow-Recognition
## 0. Introduction
This mini-project aims to identify parts of Hong Kong Road signs, especially the arrows
This project successively identify road signs of *"moving forward", "right turn", "left turn" "uturn ","no stop"*, the standard input of these networks is grayscale images of 60\*60

One fully-connected network model and two convolutional neural network model are provided. All of them have achieved more than 90% of accuracy on moderate test sets. There are some differences between this two convolutional models.

The project is based on *python 3 tensorflow Framework*, and has made use of the following libraries:

Numpy, Tensorflow, Matplotlib, scipy, PIL, pandas


## 1. Components

Conv.ipynb: 

    Notebook for convolutional networks.

Conv_util.py:

    Functions needed for the convolutional networks notebook

Full_connect.ipynb:

    Notebook for fully-connected networks.

Full_connect_util.py:
    
    Functions needed for the fully-connected networks notebook

000Readme.docx: 

    Readme in Chinese.

Teacherset.csv:

    Dataset provided by the lecturers Dr Fung in EE department, HKPU. The dataset consists of 469 100\*100 grayscale images.
    
    Each line in the csv consists of 10001 number, the first of which is the label the reset is the corresponding image data. 
    
Test.csv:

    The test set created by partner Wu Han Xi exchange from Harbin Institute of Technology using Matlab. 
    
    The test set consists of 155 60\*60 grayscale images. Each line in the csv consists of 3601 numbers.    
    
Train.csv:

    Training set also created by Wu with Matlab. consists of 5491 images.
    
    The set was created by data augmentation from several hundreds of pictures made in PPT or downloaded from the Internet.

Some Problem with the datasets: 

    ①. For *"moving forward", "right turn", "left turn" "uturn "no stop""*, the label of Wu is 1,2,3,4,5 respectively while the teacherset does not have "stop" and label the picture as 1,3,2,4.

    ②. The data from the Dr Fung comes from python while data from Wu comes from Matlab. Some rules are different in two software and the ways to read data from them are different.

    ③. The project is done in HK, as a result, the "Uturn" is turning to the right, while is turning left in the mainland. All datasets are consist of Uturn in HK version.


#The following only available on Baidu Net Disk.

checkpointset:

    Save the checkpoint from Tensorflow

origincheckpointset:

    Backup checkpoints that perform well

image ：

    pictures with names in the form of '#.jpg' that are saved here can be tested with function written in each util.py. The image.rar are original pictures that used to build train.csv and test.csvv
