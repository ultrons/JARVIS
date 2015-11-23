"""
FileName: convNet.tf.py
Abstract: Basic convolutional neural network framework
Author  : Vaibhav Singh


Attribution:
------------
    1. Inspired from: Google Tensor Flow, Advanced MNIST Tutorial
       Simplified removing second conv/pool layers
    2. DataSet Passing adapted from common load data Utility
"""
from sklearn.utils import shuffle
import loadData as ld
preTrainedVecFiles=['/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/vectors.6B.100d.splitted.aaa']
DS = { 'sst':
            # (phraseFile,labelFile) 
            ('/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleDictionary.txt',
             '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleLables.txt'),
       'mr':
           #(Neg,Pos)
           ('/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.neg',
            '/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.pos'
           )
          }

pv=ld.preTrainedVectors(preTrainedVecFiles)
format='sst'
mini_batch_size=100
epochs=20
maxwords=60
wvDim=100
numClasses=5
numFeatureMaps=20
fcSize=200
dataSet=ld.corpus(pv,DS[format],format)
trainSet,testSet,valSet=dataSet.createSplit()
trainSetSize=trainSet[1].shape[0]
numBatch=int(trainSetSize/mini_batch_size)


# Import tensorflow as usual
import tensorflow as tf

# Creating interactive session
# In order to create graph-compute-graph cycles
# in non interactive session we need to first create the graph before performing
# any computations
sess = tf.InteractiveSession()

# Input data x and True label y_ are placeholders in the computation graph as
# usual, Remember for any computation graph there are parameters of the
# computation, e.g. constants, co-efficients, some of them are computed by the
# graph, some are constants. Ones computed by graph, e.g. using some iterative 
# algorithm are variables (states), ones fixed are obviously constants, and ones which
# are fed in, are defined as placeholders (stateless)
# Theano achieves using 'given' attribute
x = tf.placeholder('float', shape=[None, maxwords*wvDim])
y_ = tf.placeholder('float', shape=[None, numClasses])


# Fun begins here
# Creating methods W and b tensor creation 
# Note we are not initializing W's with constant 0
# In order to avoid dead neurons.
# However, we are using ReLU(Rectified linear unit) neurons
# In the zero region if I remember well the gradient is non-zero
# So I have not really understood why we can't initiaize at 0,
# Need to do some gradient calculation on paper
# TO BE UPDATED....
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial= tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# defining general purpose convolutional and pooling layer
# With stride size 1 and uniform zero padding
# There output of convolution layer is of same size as inputs
# Notice max pooling is done over 2x2 grid
# This where the zoom out happens, ie we reduce the image size
# so from 28x28 we come down to 14x14
# As you would see at the second poling layer we reduce to 7x7
# Notice that convolution layer is generic enough
# to choose Number of filters i.e. number of feature maps
# Size of each feature map will be decided based on 
# stride, filter size, and input size
# Later two parameters are obvious from shapes of W and x inputs
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Defining First set of convolution and pooling layer
# Notice first two dimensions of W are size of patch filters are going to be
# looking at, the last one is Number of filter types or feature maps(
# 32), is it the cave symbols count :P
# We will comment about the third parameter when we look at the second
# For now we remember that it's the number of input channels
# convolution layer
W_conv1 = weight_variable([5,5,1,numFeatureMaps])
# Bias term is obviously of the dimension (count of neurons in a filter)
b_conv1 = bias_variable([numFeatureMaps])

# Now we are about to construct the convolutional and pooling layers
# but before we do that we need to reshape our 784 dimensional vector to 
# 28x28 grid, so that filtering over 5x5 patches could make sense
# otherwise we need to define our filters differently
# Role of first and last dimension of the tensor is not clear to me yet
# TO BE UPDATED ...
x_image = tf.reshape(x,[-1, maxwords,wvDim, 1])


# Finally we create our first set of convolution and pooling layers
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Now what does the following dimensions mean
# We can explain all, 5x5 , it is the patch size as usual
# Notice that first convolution layer computed 32 features from the input image
# what is second conv+pool layer looking at 
# It is looking at images of 14x14 dimension
# Notice I said image'S', because there are 32 such images
# because we created 32 feature maps in first set of layers
# Therefore for the second convolution layer the number of input channels is 32
# And this time around we are going to create 64 feature 
#W_conv2 = weight_variable([5,5, 32, 64])
# Bias dimension is as usual
#b_conv2 = bias_variable([64])

# Creating second set of convolution and max pooling
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

# Constructing fully connected layer
# Notice we are down to 64  7x7 images now
# 1024 is an arbitrary choice of outputs from fully connected layer
W_fc1 = weight_variable([maxwords*wvDim*numFeatureMaps/4, fcSize])
b_fc1 = bias_variable([fcSize])


# Before we enter to full connected layer we come back to vector form
#h_pool2_flat = tf.reshape(h_pool2, [-1, maxwords*wvDim*64/16])
h_pool2_flat = tf.reshape(h_pool1, [-1, maxwords*wvDim*numFeatureMaps/4])

# Choosing relu in the fully connected layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop out regularization
# From computation graph point of view this can be viewed as a layer
# To keep dropout configurable we create "Keep probability variable as place
# holder
keep_prob = tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

# Finally Weights and biases for output softmax layer
W_fc2 = weight_variable([fcSize, numClasses])
b_fc2 = bias_variable([numClasses])
# Output sofmax layer
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# Define update node with the choice of optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define nodes for precison computation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Let the game begin :)

# Initialize all variables
sess.run(tf.initialize_all_variables())

# Perform Training
for e in range(epochs):
    for i in range(numBatch):
        batch = (trainSet[0][i*mini_batch_size:(i+1)*mini_batch_size],
                 trainSet[1][i*mini_batch_size:(i+1)*mini_batch_size])
        batch = shuffle(batch[0], batch[1])
        #print "INFO", batch[0].shape, type(batch[0]), batch[1].shape, type(batch[1])
        # Print training accuracy at every 100th iteration
        if i%100 == 0: 
            train_accuracy = accuracy.eval(feed_dict={
              x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "Epoch: %d step %d, training accuracy %g"%(e, i, train_accuracy)
        #valid_accuracy = accuracy.eval(feed_dict={
        #    x:valSet[0], y_: valSet[1], keep_prob: 1.0})
        #print "                 , Validation accuracy %g"%(valid_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    # Printing Test Accuracy after each Epoch
    print "Epoch: %d, test accuracy %g"%(e, accuracy.eval(feed_dict={
        x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
