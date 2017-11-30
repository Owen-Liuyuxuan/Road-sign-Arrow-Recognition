

import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd

def get_X_and_Y(file_name):
    df = pd.read_csv(file_name)
    df = np.array(df);
    m = df.shape[0];
    X_train = np.zeros((3600,m))
    Y_train = np.zeros((5,m));
    Y_index = df[:,0]-1;
    for i in range(m):
        Y_train[Y_index[i]][i] = 1
        X_train[:,i] = df[i,1:];
    print(X_train.shape)
    print(Y_train.shape)
    return X_train/256,Y_train,Y_index
def read_teacher(file_name):
    df = pd.read_csv(file_name)
    df = np.array(df);
    m = df.shape[0];
    X = np.zeros((3600,m))
    Y = np.zeros((5,m));
    Y_index = df[:,0] - 1;
    #改变 Y_temp 适应训练集
    Y_index[Y_index==2] = 5
    Y_index[Y_index==3] = 2
    Y_index[Y_index==5] = 3;
    for i in range(m):
        Y[Y_index[i]][i] = 1;
        X[:,i] = scipy.misc.imresize(df[i,1:].reshape([100,100]),size = (60,60)).T.reshape(3600)/256;
    print(X.shape)
    print(Y.shape)
    return X,Y,Y_index

def initialize_weight(layer_dims):
    w1 = tf.get_variable('W1',[layer_dims[1],layer_dims[0]],initializer = tf.contrib.layers.xavier_initializer())
    B1 = tf.get_variable('b1',[layer_dims[1],1],initializer = tf.zeros_initializer());
    w2 = tf.get_variable('W2',[layer_dims[2],layer_dims[1]],initializer = tf.contrib.layers.xavier_initializer())
    B2 = tf.get_variable('b2',[layer_dims[2],1],initializer = tf.zeros_initializer());
    w3 = tf.get_variable('W3',[layer_dims[3],layer_dims[2]],initializer = tf.contrib.layers.xavier_initializer())
    B3 = tf.get_variable('b3',[layer_dims[3],1],initializer = tf.zeros_initializer());
    parameters = {"W1": w1,"b1": B1,"W2": w2,"b2": B2,"W3": w3,"b3": B3}
    return parameters

def forward_propagation(X1,parameters,keep_prob,keep_prob2 = 0.85):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    x1 = tf.nn.dropout(X1,keep_prob2);
    Z1 = tf.add(b1,tf.matmul(W1,x1))
    A1 = tf.nn.relu(Z1)                                            
    a1 = tf.nn.dropout(tf.nn.l2_normalize(A1,0), keep_prob);
    
    Z2 = tf.add(b2,tf.matmul(W2,a1))                                           
    A2 = tf.nn.relu(Z2)                                              
    
    a2 = tf.nn.dropout(tf.nn.l2_normalize(A2,0),keep_prob);
    
    Z3 = tf.add(b3,tf.matmul(W3,a2))
    return Z3


def random_mini_batches(X, Y, mini_batch_size = 64):


    
    (n_x, m) = X.shape                    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]


    num_complete_minibatches = math.floor(m/mini_batch_size)
    mini_batches = [None] * num_complete_minibatches;
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches[k] = mini_batch;
    

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(X_train, Y_train,layer_dims, learning_rate = 0.0001,num_epochs = 1500, minibatch_size = 32,keep_prob = 0.7,lamda = 0):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = [None]*num_epochs                                       # To keep track of the cost
    X1 = tf.placeholder(tf.float32,[n_x,None],name = 'X1')
    Y1 = tf.placeholder(tf.float32,[n_y,None],name = 'Y1')


    parameters = initialize_weight(layer_dims)

    Z3 = forward_propagation(X1, parameters,keep_prob)

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0. 
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            temp_num = 0;
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch;
                temp_num = 0;
                temp_num += 1; 
                if(temp_num%2 == 0):
                    X_temp_batch = 1-minibatch_X
                else:
                    X_temp_batch = minibatch_X;
                X_temp_batch = (1+ 0.1*np.random.randn(1)) * X_temp_batch;

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X1: minibatch_X, Y1: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 10 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs[epoch] = minibatch_cost
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        print ("Parameters have been trained!")
        string = '1'
        string = input('Want to save this session? (y/n)')
        if(string.startswith('y')):
            saver = tf.train.Saver()
            string = input('input checkpoint name');
            if (string == ''):
                string = 'default';
            model_path ='checkpointset\\'+string+'.ckpt';
            save_path = saver.save(sess, model_path);


def continue_train_with_ckpt(X_train, Y_train,checkpoint_filepath,layer_dims, learning_rate = 0.0001,num_epochs = 1500, minibatch_size = 32, print_cost = True,lamda = 0,keep_prob = 0.7):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = [None]*num_epochs                                       # To keep track of the cost
    X1 = tf.placeholder(tf.float32,[n_x,None],name = 'X1')
    Y1 = tf.placeholder(tf.float32,[n_y,None],name = 'Y1')


    parameters = initialize_weight(layer_dims)

    Z3 = forward_propagation(X1, parameters,keep_prob)

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        
        saver=tf.train.Saver()
        saver.restore(sess,checkpoint_filepath);

        for epoch in range(num_epochs):

            epoch_cost = 0. 
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            temp_num = 0;
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch;
                temp_num += 1; 
                if(temp_num%2 == 0):
                    X_temp_batch = 1-minibatch_X
                else:
                    X_temp_batch = minibatch_X;
                X_temp_batch = (1+ 0.1*np.random.randn(1)) * X_temp_batch;

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X1: minibatch_X, Y1: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 10 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs[epoch] = minibatch_cost
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        string = '1'
        string = input('Want to save this session? (y/n)\n')
        if(string.startswith('y')):
            saver = tf.train.Saver()
            string = input('input checkpoint name:\n');
            if (string == ''):
                model_path = checkpoint_filepath;
                print('Update to original file\n')
            else:
                model_path =  'checkpointset\\'+string+'.ckpt';
                print('save to '+string+'.ckpt successfully\n')
            save_path = saver.save(sess, model_path);
        tempstring = input('Want to retrain with the same setting? (input \'y\' to continue the training)\n');
        


def tensorflow_evaluate(checkpoint_filepath,X,Y_original,layer_dims,printprediction = False,printaccuracy = True):
    ops.reset_default_graph()
    (n_x, m) = X.shape
    X1 = tf.placeholder(tf.float32,[n_x,None],name = 'X1')
    parameters = initialize_weight(layer_dims)
    Z3 = forward_propagation(X1, parameters,1,1)
    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess,checkpoint_filepath);
        output = sess.run(Z3, feed_dict={X1: X})
        predictions = np.argmax(output,axis = 0);
        correct_prediction =(predictions == Y_original)
        if (printprediction):
            print(predictions+1)
        if (printaccuracy):
            print(correct_prediction.sum()/m)
        
def read_one_picture_to_2darray(picname = '0',showpic = True):
    fname = "image/"+picname+".jpg"
    image = Image.open(fname)
    image1 = image.convert('L')
    my_image = np.array(image1)
    my_image = scipy.misc.imresize(my_image,size = (60,60))
    if (showpic):
        image2 = Image.fromarray(my_image);
        image2.show();
    return my_image.T

def one_picture_to_1darray(picname = '0',showpic = True):
    my_image = read_one_picture_to_2darray(picname,showpic);
    my_image = my_image.reshape(3600);
    my_image = my_image/256
    return my_image

def read_pictures_and_evaluate(checkpoint_filepath,layer_dims,picture_num = 4,kind = 1,picture_start_num = 0,
                               showpic = False,printprediction = True,printaccuracy = False):
    X = np.zeros((3600,picture_num))
    for i in range(picture_num):
        X[:,i] = one_picture_to_1darray(str(i+picture_start_num),showpic)
    tensorflow_evaluate(checkpoint_filepath,X,kind*np.ones(picture_num),layer_dims,printprediction,printaccuracy)
    


