
# coding: utf-8

# ## Convolutional Network

# Load the libraries and data

# In[7]:


import numpy as np
from Conv_util import*
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd


# In[5]:


Train_file = 'Train.csv'
Test_file = 'Test.csv'
Teaching_file = "Teacherset.csv"


# In[24]:


X_train,Y_train,Y_train_index = get_X_and_Y(Train_file,Xshape = [60,60,1],Yshape = 5)


# In[33]:


X_test,Y_test,Y_test_index = get_X_and_Y(Test_file,Xshape = [60,60,1],Yshape = 5)


# In[8]:


X_teach,Y_teach,Y_teach_index = read_teacher(Teaching_file)


# model function creates a completely new model. 
# 
# After the training is finished, one can decide whether to save the new model and the name of the new model. The new model will be saved to the checkpointset in this file.

# In[ ]:


model(X_train, Y_train,learning_rate = 0.009,num_epochs = 1, minibatch_size = 64,keep_prob = 0.5,keep_prob2 = 0.6)


#  The checkname variable below defines the checkpoint being used in the following functions
#  
#  Notice that here we have two check point available.
#  
#  They are different in how the first dropout is applied.
#  
#  The default network structure is suitable for the 'hidden' one.
# 
#  If one wants to train in the 'dropout_in_input'setting, the forward_prob function in Conv_util.py has to be adjusted

# In[42]:


checkpoint_name = 'Conv_with_dropout_in_hidden_layer2'
#checkpoint_name = 'Conv_with_dropout_in_input_layer'
checkpoint_filepath = 'checkpointset\\'+checkpoint_name+'.ckpt'


# Continue_train_with ckpt continues to train the checkpoint in that checkpoint_file defined.
# 
# When asked save where to save the ckpt, press 'Enter' can overwrite the ckpt just used by default.
# 
# If one would not want to overwrite the ckpt given but mis-overwrite the ckpt, reload them from the dic:'origincheckpointset'
# 
# When asked whether to train again, press 'Enter' will not start the retraining.

# In[ ]:


continue_model(X_train, Y_train,checkpoint_name,learning_rate = 0.0015,
          num_epochs = 20, minibatch_size = 32,keep_prob = 0.5,keep_prob2 = 0.85)


# Evaluate how well the model has been doing on different sets of csv files using the following function. Notice the changes due to the flip in colours.

# In[43]:


tensorflow_evaluate(checkpoint_filepath,X_train,Y_train_index,print_prediction = False)


# In[44]:


tensorflow_evaluate(checkpoint_filepath,1-X_train,Y_train_index,print_prediction = False)


# In[45]:


tensorflow_evaluate(checkpoint_filepath,X_test,Y_test_index,print_prediction = False)


# In[46]:


tensorflow_evaluate(checkpoint_filepath,1-X_test,Y_test_index,print_prediction = False)


# In[47]:


tensorflow_evaluate(checkpoint_filepath,X_teach,Y_teach_index,print_prediction = False)


# In[48]:


tensorflow_evaluate(checkpoint_filepath,1-X_teach,Y_teach_index,print_prediction = False)


# In image recognition, maybe we should be careful about dropout in the input layer, especially when symmetrical white and black contents are expected because dropout is creating asymmetry.

# Evaluta how well the model has been doing on the image collected in the image pack.
# 
# This function can read image from 'pic_start_num.jpg' to 'pic_start_num+pic_num-1.jpg' and give the calculated output
# 
# If 'showpic = True', then the function will show pictures in 60\*60 grey scale, and it will be rather slow.

# In[ ]:


read_pictures_and_evaluate(checkpoint_filepath,picture_num = 4,kind = 1,picture_start_num = 0,print_prediction = True)

