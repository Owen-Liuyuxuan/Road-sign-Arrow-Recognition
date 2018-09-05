# Road-sign-Arrow-Recognition
Complete Version of the mini-project file can be downloaded directly from since files over 25M are not accepted in Github:

        https://pan.baidu.com/s/1sl6xGzr#list/path=%2F
        
Notice: 

        currently only the 'train.csv' and 'image.rar' are up to date in this Baidu Net Disk. // 11/30/2017
## 0. Introduction
This mini-project aims to identify parts of Hong Kong Road signs, especially the arrows.

This project successively identify road signs of *"moving forward", "right turn", "left turn" "uturn ","no stop"*, the standard input of these networks is grayscale images of 60\*60.

One fully-connected network model and two convolutional neural network model are provided. All of them have achieved more than 90% of accuracy on moderate test sets. There are some differences between this two convolutional models.

The project is based on *python 3 tensorflow Framework*, and has made use of the following libraries:

Numpy, Tensorflow, Matplotlib, scipy, PIL, pandas

Liu Yu Xuan and Wu Han Xi contribute equally in the project.


## 1. Components

Conv.ipynb: 

    Notebook for convolutional networks.

Conv_util.py:

    Functions needed by the convolutional networks notebook

Full_connect.ipynb:

    Notebook for fully-connected networks.

Full_connect_util.py:
    
    Functions needed by the fully-connected networks notebook

000Readme.docx: 

    Readme in Chinese.

Teacherset.csv:

    Dataset provided by the lecturers Dr Fung in EE department, HKPU. The dataset consists of 469 100\*100 grayscale images.
    
    Each line in the csv file consists of 10001 number, the first of which is the label the rest is the corresponding image data. 

Teachertest.csv:
     
     Another dataset provided by Dr Fung. He originally expects us to train with 'Teacher.csv' and test with 'Teachertest.csv'.
     
     In this project, we take both file as final test set.
    
Test.csv:

    The test set created using Matlab. 
    
    The test set consists of 155 60\*60 gray-scale images. Each line in the csv file consists of 3601 numbers.    
    
    The test set has been considered as cross-validation set for this project.
    

Some Problem with the datasets: 

    ①. For "moving forward", "right turn", "left turn" "uturn ","no stop", the label from Wu is 1,2,3,4,5 respectively while the teacherset.csv does not have "no stop" and label the picture as 1,3,2,4.

    ②. The data from Dr Fung comes from python while data from Wu comes from Matlab. Some rules are different in two software and the ways to read data from them are different.

    ③. The project is done in HK, as a result, the "Uturn" is turning to the right, while is turning left in the mainland. All datasets just include "Uturn" in HK version.

image ：

    pictures with names in the form of '#.jpg' that are saved here can be tested with function written in each util.py.

checkpointset:

    Save the checkpoints from Tensorflow. There are three possible model inside. They are not perfect, but at least respectable.


# The following only available on Baidu Net Disk.

Train.csv:

    Training set also created with Matlab, consists of 5491 images.
    
    The set was created by data augmentation from several hundreds of pictures made in PPT or downloaded from the Internet.


origincheckpointset:

    Backup checkpoints that perform well

image.rar:

        The image.rar are original pictures that used to build train.csv and test.csv
