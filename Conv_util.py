import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import pandas as pd

def get_X_and_Y(training_file,Xshape = [60,60,1],Yshape = 5):
    df = pd.read_csv(training_file)
    df = np.array(df);
    m = df.shape[0];
    X_train = np.zeros((m,Xshape[0],Xshape[1],Xshape[2]))
    Y_train = np.zeros((m,Yshape));
    Y_index = df[:,0]-1;
    for i in range(m):
        Y_train[i][Y_index[i]] = 1
        X_train[i] = df[i,1:].reshape([Xshape[0],Xshape[1],1])
    print(X_train.shape)
    print(Y_train.shape)
    return X_train/256,Y_train,Y_index

def read_teacher(file_name):
    df = pd.read_csv(file_name)
    df = np.array(df);
    m = df.shape[0];
    X = np.zeros((m,60,60,1))
    Y = np.zeros((m,5));
    Y_index = df[:,0] - 1;
    #改变 Y_temp 适应训练集
    Y_index[Y_index==2] = 5
    Y_index[Y_index==3] = 2
    Y_index[Y_index==5] = 3;
    for i in range(m):
        Y[i][Y_index[i]] = 1;
        X[i,:,:,0] = scipy.misc.imresize(df[i,1:].reshape([100,100]),size = (60,60)).T;
    print(X.shape)
    print(Y.shape)
    return X/256,Y,Y_index

def random_mini_batches(X, Y, mini_batch_size = 64):

    (m, n_H0, n_W0, n_C0) = X.shape    
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    mini_batches = [None] * num_complete_minibatches;
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches[k] = mini_batch;
    

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_weight():
    W1 = tf.get_variable("W1",[3,3,1,8],initializer = tf.contrib.layers.xavier_initializer())  
    W2 = tf.get_variable('W2',[5,5,8,24],initializer = tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable('W3',[5,5,24,48],initializer = tf.contrib.layers.xavier_initializer())
    parameters = {"W1": W1,"W2": W2,"W3": W3}
    return parameters
def forward_propagation(X,parameters,keep_prob,keep_prob2 = 0.85):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    #x1 = tf.nn.dropout(X,keep_prob2)
    #Z1 = tf.nn.conv2d(x1,W1, strides = [1,1,1,1], padding = 'SAME');
    #A1 = tf.nn.relu(Z1)
    #P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') 
    #Z2 = tf.nn.conv2d(P1,W2,strides = [1,1,1,1] , padding = 'SAME')
    
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME');
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') 
    p1 = tf.nn.dropout(P1,keep_prob2)
    Z2 = tf.nn.conv2d(p1,W2,strides = [1,1,1,1] , padding = 'SAME')
    
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')
    
    Z3 = tf.nn.conv2d(P2,W3,strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME');
    
    F3 = tf.contrib.layers.flatten(P3)
    d3 = tf.nn.dropout(F3,keep_prob);
    
    F4 = tf.contrib.layers.fully_connected(d3,60);#11_25 original 60
    d4 = tf.nn.dropout(F4,keep_prob);
    
    F5 = tf.contrib.layers.fully_connected(d4,40)
    d5 = tf.nn.dropout(F5,keep_prob);
    
    F6 = tf.contrib.layers.fully_connected(d5,5,activation_fn = None)
    
    return F6

def model(X_train, Y_train,learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64,keep_prob = 1,keep_prob2 = 1):    
    ops.reset_default_graph()                      
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = [None]*num_epochs                                       
    
    X = tf.placeholder(tf.float32,(None, n_H0, n_W0, n_C0),name = 'X')
    Y = tf.placeholder(tf.float32,(None,n_y),name = 'Y')
    
    parameters = initialize_weight()
    
    F6 = forward_propagation(X,parameters,keep_prob,keep_prob2)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = F6,labels = Y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train setZ
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            temp_num = 0;
            for minibatch in minibatches:
                temp_num += 1; 
                (minibatch_X, minibatch_Y) = minibatch
                if(temp_num%2 == 0):
                    X_temp_batch = 1-minibatch_X
                else:
                    X_temp_batch = minibatch_X;
                X_temp_batch = (1+ 0.1*np.random.randn(1)) * X_temp_batch;
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: X_temp_batch, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            if epoch % 5 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs[epoch] = minibatch_cost
    
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
            model_path = 'checkpointset\\'+string+'.ckpt';
            save_path = saver.save(sess, model_path);



def continue_model(X_train, Y_train,checkpoint_name,learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64,keep_prob = 1,keep_prob2 = 1):
    
    ops.reset_default_graph()                      
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = [None]*num_epochs                                       
    
    X = tf.placeholder(tf.float32,(None, n_H0, n_W0, n_C0),name = 'X')
    Y = tf.placeholder(tf.float32,(None,n_y),name = 'Y')
    
    parameters = initialize_weight()
    
    F6 = forward_propagation(X,parameters,keep_prob,keep_prob2)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = F6,labels = Y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    
    with tf.Session() as sess:
        
        #take back all the parameter in the checkpoint, and then train upon it.
        saver=tf.train.Saver()
        checkpoint_filepath = 'checkpointset\\'+checkpoint_name+'.ckpt'
        saver.restore(sess,checkpoint_filepath);
        
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            #this function shuffle the traing set 
            #and return a list of minibatch
            temp_num = 0;
            for minibatch in minibatches:
                temp_num += 1; 
                (minibatch_X, minibatch_Y) = minibatch
                # each minibatch is a tuple, each cell of a tuple is a list, each list is a training sample;
                
                # The following part of the function try to add noise to picture and make some pictures flip color.
                if(temp_num%2 == 0):
                    X_temp_batch = 1-minibatch_X
                    
                else:
                    X_temp_batch = minibatch_X;
                X_temp_batch = (1+ 0.1*np.random.randn(1)) * X_temp_batch;
                
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: X_temp_batch, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches

            if epoch % 5 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs[epoch] = minibatch_cost
        
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        #decide whether to save the checkpoint.
        print ("Parameters have been trained!")
        string = '1'
        string = input('Want to save this session? (y/n)')
        if(string.startswith('y')):
            saver = tf.train.Saver()
            string = input('input checkpoint name');
            if (string == ''):
                string = checkpoint_name;
            model_path = 'checkpointset\\'+string+'.ckpt';
            save_path = saver.save(sess, model_path);


def tensorflow_evaluate(checkpoint_filepath,X_sample,Y_original,print_prediction = False,print_accuracy = True):
    ops.reset_default_graph()                      
    (m, n_H0, n_W0, n_C0) = X_sample.shape
    X = tf.placeholder(tf.float32,(None, n_H0, n_W0, n_C0),name = 'X')
    parameters = initialize_weight()
    F6 = forward_propagation(X,parameters,1,1)
    O4 = tf.nn.softmax(F6);
    predict = []
    
    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess,checkpoint_filepath);
        correct_prediction = 0;
        for i in range(m):
            output = sess.run(O4, feed_dict={X: X_sample[i,:,:,:].reshape([1,60,60,1])})
            predictions = np.argmax(output,axis = 1);
            predict += predictions.tolist()
            correct_prediction += (predictions[0] == Y_original[i]);
    if(print_prediction):
        print(np.array(predict,dtype = 'float32')+1)
    if(print_accuracy):
        print('Accuracy: ' + str(correct_prediction/m))
        print('Correct number: '+ str(correct_prediction) )



def read_one_picture_to_2darray(picname = '0',showpic = True):
    #the picture should can be any size even RGB in jpg
    fname = "image/"+picname+".jpg"
    image = Image.open(fname)
    image1 = image.convert('L')
    my_image = np.array(image1)
    my_image = scipy.misc.imresize(my_image,size = (60,60))
    if (showpic):
        image2 = Image.fromarray(my_image);
        image2.show();
    #Transpose the picture because the difference between Matlab and python
    my_image = my_image.T
    my_image = my_image.reshape([60,60])/256
    return my_image

def read_pictures_and_evaluate(checkpoint_filepath,picture_num = 4,kind = 1,picture_start_num = 0,
                               showpic = False,print_prediction = True,print_accuracy = False):
    X = np.zeros((picture_num,60,60,1))
    for i in range(picture_num):
        X[i,:,:,0] = read_one_picture_to_2darray(str(i+picture_start_num),showpic)
    tensorflow_evaluate(checkpoint_filepath,X,kind*np.ones(picture_num),print_prediction ,print_accuracy)





