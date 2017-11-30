
# coding: utf-8

# # Fully_connected_neural network

# In[ ]:


import numpy as np
from Full_connect_util import*
import tensorflow as tf
from tensorflow.python.framework import ops


# In[ ]:


Train_file = 'Train.csv'
Test_file = 'Test.csv'
Teacher_file = 'Teacherset.csv'


# Layer dimension is listed here. 
# 
# This list could be modified to change the number of cells in hidden layers.

# In[ ]:


layer_dims = [3600,350,50,5]


# In[ ]:


X_train,Y_train,Y_train_index = get_X_and_Y(Train_file)


# In[ ]:


X_test,Y_test,Y_test_index= get_X_and_Y(Test_file)


# In[ ]:


X_teach,Y_teach,Y_teach_index = read_teacher(Teacher_file)


# model function creates a completely new model. 
# 
# After the training is finished, one can decide whether to save the new model and the name of the new model. The new model will be saved to the checkpointset in this file.

# In[ ]:


model(X_train, Y_train,layer_dims, learning_rate = 0.001,num_epochs = 1, minibatch_size = 64,keep_prob = 0.7)


#  The checkname variable below defines the checkpoint being used in the following functions

# In[ ]:


checkname = 'Full_connect_model'

checkpoint_filepath = 'checkpointset\\'+checkname+'.ckpt'


#  Continue_train_with ckpt continues to train the checkpoint in that checkpoint_file defined.
#  
#  When asked save where to save the ckpt, press 'Enter' can overwrite the ckpt just used by default.
#  
#  If one would not want to overwrite the ckpt given but mis-overwrite the ckpt, reload them from the dic:'origincheckpointset'
#  
#  When asked wheter to train again, press 'Enter' will not start the retraining.

# In[ ]:


continue_train_with_ckpt(X_train, Y_train,checkpoint_filepath,layer_dims, learning_rate = 0.00006,num_epochs = 1, minibatch_size = 1000,keep_prob = 0.55)


# Evaluate how well the model has been doing on different sets of csv files using the following function  

# In[ ]:


tensorflow_evaluate( checkpoint_filepath,X_train,Y_train_index,layer_dims,False)


# In[ ]:


tensorflow_evaluate( checkpoint_filepath,X_test,Y_test_index,layer_dims,False)


# In[ ]:


tensorflow_evaluate( checkpoint_filepath,X_teach,Y_teach_index,layer_dims,False)


# Evaluta how well the model has been doing on the images collected in the image pack.
# 
# This function can read image from 'pic_start_num.jpg' to 'pic_start_num+pic_num-1.jpg' and give the calculated output
# 
# If 'showpic = True', then the function will show pictures in 60\*60 grey scale, and it will be rather slow.

# In[ ]:


read_pictures_and_evaluate(checkpoint_filepath,layer_dims,picture_num = 4,picture_start_num = 0,
                               showpic = False,printprediction = True)

