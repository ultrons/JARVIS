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
# Import tensorflow as usual
import tensorflow as tf
from datetime import datetime

class tfNetwork(object):
    def __init__(self,mini_batch_size,epochs, optimizer, networkSpec,trainSet, testSet, valSet, modelFile):
        tf.set_random_seed(786)
        self.mini_batch_size=mini_batch_size
        self.epochs=epochs
        self.networkSpec=networkSpec
        self.trainSet=trainSet
        self.testSet=testSet
        self.valSet=valSet
        self.optimizer=optimizer
        self.numClasses=networkSpec['softMaxLayerDim']
        self.trainPointer=0
        self.trainSetSize=trainSet[1].shape[0]
        modelFileSuffix=str(datetime.now())
        modelFileSuffix=modelFileSuffix.replace(' ','_')
        modelFileSuffix=modelFileSuffix.replace(':','_')
        self.modelFile=modelFile+'_'+modelFileSuffix

    def setInput(self, imageX, imageY):
        self.imageX=imageX
        self.imageY=imageY
        # Input data x and True label y_ are placeholders in the computation graph as
        # usual, Remember for any computation graph there are parameters of the
        # computation, e.g. constants, co-efficients, some of them are computed by the
        # graph, some are constants. Ones computed by graph, e.g. using some iterative 
        # algorithm are variables (states), ones fixed are obviously constants, and ones which
        # are fed in, are defined as placeholders (stateless)
        # Theano achieves using 'given' attribute
        self.x = tf.placeholder('float', shape=[None, imageX*imageY])
        self.y_ = tf.placeholder('float', shape=[None, self.numClasses])
        self.x_image = tf.reshape(self.x,[-1, imageX,imageY, 1])
    # Fun begins here
    # Creating methods W and b tensor creation 
    # Note we are not initializing W's with constant 0
    # In order to avoid dead neurons.
    # However, we are using ReLU(Rectified linear unit) neurons
    # In the zero region if I remember well the gradient is non-zero
    # So I have not really understood why we can't initiaize at 0,
    # Need to do some gradient calculation on paper
    # TO BE UPDATED....
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.08, seed=786)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial= tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # defining general purpose convolutional and pooling layer
    def conv2d(self,x,W,strideX, strideY):
        return tf.nn.conv2d(x,W, strides=[1,strideX,strideY,1], padding='SAME')
    def max_pool(self,x, poolX, poolY):
        return tf.nn.max_pool(x, ksize=[1,poolX,poolY,1], strides=[1,poolX,poolX,1],
                              padding='SAME')

    def build(self):
        W_conv=[]
        b_conv=[]
        h_conv=[]
        h_pool=[]
        h_conv_r=[]
        channels=1
        commonPooling=False
        XY=[(self.imageX, self.imageY)]
        #print "HOT",  XY
        for (filterSpec, filterCount, poolX,poolY, strideX, strideY) in self.networkSpec['convPoolLayers']:
            h_pool=[]
            totalChannels=[]
            pooledSamples=0
            XY_r=[]
            if len(h_conv_r) == 0: 
                h_conv_r=[self.x_image]*len(filterSpec)
                channels=[1]*len(filterSpec)
                XY=[(self.imageX, self.imageY)]*len(filterSpec)

            for filter_id, (filterX,filterY) in enumerate(filterSpec):
                print "HIT", filterX,filterY,channels[filter_id],filterCount
                W_conv.append(self.weight_variable([filterX,filterY,channels[filter_id],filterCount]))
                b_conv.append(self.bias_variable([filterCount]))
                h_conv.append(tf.nn.relu(self.conv2d(h_conv_r[filter_id], W_conv[-1], strideX,
                                           strideY) + b_conv[-1]))
                h_pool.append(self.max_pool(h_conv[-1], poolX, poolY))
                totalChannels.append(filterCount)
                #print "HIT",  XY[-1][0]/strideX,filterX, XY[-1][1]/strideY,filterY
                XY_r.append(((XY[filter_id][0]/strideX/poolX),(XY[filter_id][1]/strideY/poolY)))
                print "HOT", XY_r[filter_id][0], XY_r[filter_id][1], filterCount
                pooledSamples+=XY_r[filter_id][0]*XY_r[filter_id][1]*filterCount
               # print pooledSamples
            h_conv_r=h_pool
            channels=totalChannels
            XY=XY_r
        
        h_pool_r=[tf.reshape(x, [-1]) for x in h_pool]
        fullyConnectedLayerDim=self.networkSpec['fullyConnectedLayerDim']
        W_fc1 = self.weight_variable([pooledSamples, fullyConnectedLayerDim])
        b_fc1 = self.bias_variable([fullyConnectedLayerDim])
        # Before we enter to full connected layer we come back to vector form
        h_pool2_flat = tf.reshape(tf.concat(0, h_pool_r), [-1, pooledSamples])
        
        # Choosing relu in the fully connected layer
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # drop out regularization
        # From computation graph point of view this can be viewed as a layer
        # To keep dropout configurable we create "Keep probability variable as place
        # holder
        self.keep_prob = tf.placeholder('float')
        h_fc1_drop=tf.nn.dropout(h_fc1, self.keep_prob)
        
        # Finally Weights and biases for output softmax layer
        W_fc2 = self.weight_variable([fullyConnectedLayerDim, self.numClasses])
        b_fc2 = self.bias_variable([self.numClasses])
        # Output sofmax layer
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    def nextBatch(self):
        if self.trainPointer >= self.trainSetSize: 
            self.trainPointer = 0
        b = shuffle(
            self.trainSet[0][self.trainPointer:self.trainPointer + self.mini_batch_size],
            self.trainSet[1][self.trainPointer:self.trainPointer + self.mini_batch_size]
            )
        self.trainPointer+=self.mini_batch_size
        return b,self.trainPointer
    def computeAcuracy(self, data):
        Pointer=0
        batch_size=700
        accData=[]
        while Pointer < data[1].shape[0]:
            batch=(
                data[0][Pointer:Pointer+batch_size],
                data[1][Pointer:Pointer+batch_size]
            )
            Pointer+=batch_size
            accData.append(self.accuracy.eval(feed_dict={self.x:batch[0],
                                                           self.y_: batch[1],
                                                           self.keep_prob: 1.0}))
        return sum(accData)/len(accData)




    def train(self):
        # define cost function
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        # Define update node with the choice of optimizer
        if self.optimizer == 'ADAM': 
           train_step = tf.train.AdamOptimizer(
               learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=5e-3
               ).minimize(cross_entropy)
        if self.optimizer == 'ADAGRAD': 
           train_step = tf.train.AdagradOptimizer(
               learning_rate=0.00001).minimize(cross_entropy)
        #train_step=tf.train.GradientDescentOptimizer(0.00015).minimize(cross_entropy)
    
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        #correct_prediction = tf.equal(tf.cast((self.y_conv>1), 'float'),
        #                              tf.cast(self.y_, 'float'))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        
        # Perform Training
        for e in range(self.epochs):
            bestAccuracy=0
            self.trainSet = shuffle(self.trainSet[0], self.trainSet[1])
            for i in range(int(self.trainSetSize/self.mini_batch_size)):
                batch, batchNumber = self.nextBatch()
                if batch[1].shape[0] == 0: 
                    print "Warning Empty batch !!!!", i, "skipping!!!!"
                    continue
                
                train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1],
                                          self.keep_prob: 0.5})
                #f = int(self.trainSetSize*0.1/self.mini_batch_size) == 0
                #if f == 0: f = 1
                f=100
                if i % f == 0 :
                    train_accuracy = self.accuracy.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print "VA estimate Start.."
                    validation_accuracy=self.computeAcuracy(self.valSet)
                    print "Batch %d: TRaining Accuracy: %g, Validation Accuracy:%g " %(i,train_accuracy, validation_accuracy)
                    if validation_accuracy >= bestAccuracy:
                        modelFile=self.modelFile+'_E'+str(e)+'_B'+str(i)
                        save_path = saver.save(sess, modelFile)
                        print "Model saved in file: ", save_path
                        
                        print "VA comparison Start.."
                        bestAccuracy = validation_accuracy
                        print "TA estimate Start.."
                        test_accuracy=self.computeAcuracy(self.testSet)
            print "Epoch: %d , best Validation Accuracy: %g, corresponding test Accuracy %g" %(e, bestAccuracy, test_accuracy)
        sess.close()
