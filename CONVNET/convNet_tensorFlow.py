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
    def __init__(self,mini_batch_size,epochs, optimizer, networkSpec,trainSet,
                 testSet, valSet, modelFile, startPoint, normMax):
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
        self.startPoint=startPoint
        self.normMax=normMax

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
        with tf.name_scope("INPUTS"):
            self.x = tf.placeholder('float', shape=[None, imageX*imageY],
                                    name="1D_input_vector" )
            self.y_ = tf.placeholder('float', shape=[None, self.numClasses], name="input_label")
            self.x_image = tf.reshape(self.x,[-1, imageX,imageY, 1], name="2D_input_image")
    # Fun begins here
    # Creating methods W and b tensor creation 
    # Note we are not initializing W's with constant 0
    # In order to avoid dead neurons.
    # However, we are using ReLU(Rectified linear unit) neurons
    # In the zero region if I remember well the gradient is non-zero
    # So I have not really understood why we can't initiaize at 0,
    # Need to do some gradient calculation on paper
    # TO BE UPDATED....
    def weight_variable(self,shape, id):
        return tf.get_variable("weights_%s" %id, shape,
                               initializer=tf.random_normal_initializer(mean=0,
                                                                        stddev=0.001,
                                                                       seed=786))
    def bias_variable(self,shape, id):
        return tf.get_variable("bias_%s" %id, shape, initializer=tf.constant_initializer(0.1))

    def conv2d(self,x,W,strideX=1, strideY=1,padding='SAME'):
        return tf.nn.conv2d(x,W, strides=[1,strideX,strideY,1], padding=padding,
                           name="CONVOLVE")
    def max_pool(self,x, poolX, poolY):
        return tf.nn.max_pool(x, ksize=[1,poolX,poolY,1], strides=[1,poolX,poolY,1],
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
        for layerid, (filterSpec, filterCount, poolX,poolY, strideX, strideY) in enumerate(self.networkSpec['convPoolLayers']):
            h_pool=[]
            totalChannels=[]
            pooledSamples=0
            XY_r=[]
            if len(h_conv_r) == 0: 
                h_conv_r=[self.x_image]*len(filterSpec)
                channels=[1]*len(filterSpec)
                XY=[(self.imageX, self.imageY)]*len(filterSpec)
            with tf.name_scope("CONV_POOL_%d" %layerid):
                for filter_id, (filterX,filterY) in enumerate(filterSpec):
                    with tf.name_scope("FILTER_%dx%d" %(filterX,filterY)):
                        W_conv.append(self.weight_variable([filterX,filterY,channels[filter_id],filterCount],
                                                               "FILTER_%dx%d" %(filterX,filterY)))
                        b_conv.append(self.bias_variable([filterCount], "FILTER_%dx%d" %(filterX,filterY)))
                        #h_conv.append(tf.nn.tanh(self.conv2d(h_conv_r[filter_id], W_conv[-1], strideX,
                        #                           strideY) + b_conv[-1]))
                        h_conv.append(tf.nn.relu(self.conv2d(h_conv_r[filter_id], W_conv[-1], strideX,
                                                   strideY) + b_conv[-1]))
                    with tf.name_scope("POOL_%dx%d" %(poolX,poolY)):
                        h_pool.append(self.max_pool(h_conv[-1], poolX, poolY))
                    totalChannels.append(filterCount)
                    XY_r.append(((XY[filter_id][0]/strideX/poolX),(XY[filter_id][1]/strideY/poolY)))
                    pooledSamples+=XY_r[filter_id][0]*XY_r[filter_id][1]*filterCount
                h_conv_r=h_pool
                channels=totalChannels
                XY=XY_r
        
        h_pool_r=[tf.reshape(x, [-1]) for x in h_pool]
        fullyConnectedLayerDim=self.networkSpec['fullyConnectedLayerDim']
        with tf.name_scope("FULLY_CONNECTED_LAYER"):
            W_fc1 = self.weight_variable([pooledSamples,
                                          fullyConnectedLayerDim], "FC")
            b_fc1 = self.bias_variable([fullyConnectedLayerDim],  "FC")
            # Before we enter to full connected layer we come back to vector form
            h_pool2_flat = tf.reshape(tf.concat(0, h_pool_r), [-1, pooledSamples])
            # Choosing relu in the fully connected layer
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        # drop out regularization
        # From computation graph point of view this can be viewed as a layer
        # To keep dropout configurable we create "Keep probability variable as place
        # holder
        self.keep_prob = tf.placeholder('float', name="keep_prob")
        with tf.name_scope("DROPOUT"):
            h_fc1_drop=tf.nn.dropout(h_fc1, self.keep_prob)
        
        with tf.name_scope("SOFTMAX"):
            # Finally Weights and biases for output softmax layer
            W_fc2 = self.weight_variable([fullyConnectedLayerDim,
                                          self.numClasses], "softmax")
            b_fc2 = self.bias_variable([self.numClasses],"softmax")
            # Output sofmax layer
            self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            #self.weightSummary=tf.image_summary('Final_Layer_Weight',tf.reshape(W_fc2, [20, 2, 3]))
            #self.weightSummary=self._activation_summary(W_fc2)
        self.W=W_conv+[W_fc1, W_fc2]
        self.b=b_conv+[b_fc1, b_fc2]
    def _activation_summary(self,x):
        """Helper to create summaries for activations.
      
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
      
        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        #tf.histogram_summary(tensor_name + '/activations', x)
        tensor_name=x.op.name
        tf.histogram_summary(tensor_name + '/activations', x)
        tf.scalar_summary(tensor_name + '/weights', tf.nn.zero_fraction(x))



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
        #batch_size=data[1].shape[0]
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

    def clipByNorm(self,s):
        for t in self.W:
            t=tf.clip_by_norm(t, s)
        for t in self.b:
            t=tf.clip_by_norm(t, s)

    def printEval(self, ctype='fineGrained'):
        if ctype == 'fineGrained':
            correct_prediction = tf.equal(tf.argmax(self.y_conv,1),
                                          tf.argmax(self.y_,1))
        else:
            binMatrix1=tf.concat(0,[tf.ones([3,1]), tf.zeros([2,1])])
            binMatrix2=tf.concat(0,[tf.zeros([2,1]), tf.ones([3,1])])
            binMatrix=tf.concat(1,[binMatrix2,binMatrix1])
            binPred=tf.matmul(self.y_conv, binMatrix)
            binLabel=tf.matmul(self.y_,binMatrix)
                            
            correct_prediction=tf.equal(tf.argmax(binPred,1), tf.argmax(binLabel,1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),
                                          name="Accuracy")

        saver = tf.train.Saver()


        #sess = tf.InteractiveSession()
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8, intra_op_parallelism_threads=8))

        with sess.as_default(): 
    
            # Initialize all variables
            if self.startPoint is None: 
                print "Please Specify a Model using -s option, to print predictions from"
                exit()
            else:
                print "Loading %s ..." %(self.startPoint)
                saver.restore(sess, self.startPoint)
            testAccuracy=self.accuracy.eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
            print 'Overall Test Accuracy: %g' %(testAccuracy)
            predictedLabel=tf.argmax(self.y_conv,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
            trueLabel= tf.argmax(self.y_,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
            print predictedLabel, trueLabel
            for i, (p,t) in enumerate(zip(predictedLabel,trueLabel)):
                print "Index: %d, Prediction: %d Truth:%d, exact Label:" %(i,p,t) , self.testSet[1][i]

    def train(self,embeddingMatrix=None,ctype='fineGrained'):
        # define cost function
        with tf.name_scope("COST"):
            cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv), name="cross_entropy") 
            ce_summ = tf.scalar_summary("cross entropy", cross_entropy)

        #cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        # Define update node with the choice of optimizer
        if self.optimizer == 'ADAM': 
           opt = tf.train.AdamOptimizer(
               learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=5e-3
               )
        if self.optimizer == 'ADAGRAD': 
           opt = tf.train.AdagradOptimizer(
               learning_rate=0.00001)

        #train_step=tf.train.GradientDescentOptimizer(0.00015).minimize(cross_entropy)
        grads = opt.compute_gradients(cross_entropy)
        apply_gradient_op = opt.apply_gradients(grads)
        for grad, var in grads: 
            if grad: tf.histogram_summary(var.op.name + '/gradients', grad)

        print "Training for %s Classification" %ctype
        if ctype == 'fineGrained':
            correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        else:
            binMatrix1=tf.concat(0,[tf.ones([3,1]), tf.zeros([2,1])])
            binMatrix2=tf.concat(0,[tf.zeros([2,1]), tf.ones([3,1])])
            binMatrix=tf.concat(1,[binMatrix2,binMatrix1])
            binPred=tf.matmul(self.y_conv, binMatrix)
            binLabel=tf.matmul(self.y_,binMatrix)
            correct_prediction=tf.equal(tf.argmax(binPred,1), tf.argmax(binLabel,1))


        with tf.name_scope("ACCURACY"): 
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),
                                          name="Accuracy")

        saver = tf.train.Saver()


        #sess = tf.InteractiveSession()
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8, intra_op_parallelism_threads=8))
        summary_writer = tf.train.SummaryWriter('./logs', sess.graph_def)
          # Add histograms for trainable variables.
        for var in tf.trainable_variables(): 
            tf.histogram_summary(var.op.name, var)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        fp=open('./labelComparison.rpt', 'w+')
        with sess.as_default(): 
    
            # Initialize all variables
            if self.startPoint is None: 
                sess.run(tf.initialize_all_variables())
            else:
                saver.restore(sess, self.startPoint)
                
            
            # Perform Training
            bestAccuracy=0
            for e in range(self.epochs):
                self.trainSet = shuffle(self.trainSet[0], self.trainSet[1])
                for i in range(int(self.trainSetSize/self.mini_batch_size)):
                    self.clipByNorm(self.normMax)
                    batch, batchNumber = self.nextBatch()
                    if batch[1].shape[0] == 0: 
                        print "Warning Empty batch !!!!", i, "skipping!!!!"
                        continue
                    
                    apply_gradient_op.run(feed_dict={self.x: batch[0], self.y_: batch[1],
                                              self.keep_prob: 0.5})
                    f = int(self.trainSetSize*0.1/self.mini_batch_size)
                    #if f == 0: f = 1
                    #f=2
                    if i % f == 0 :
                        train_accuracy = self.accuracy.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                        validation_accuracy=self.computeAcuracy(self.valSet)
                        print "Batch %d: TRaining Accuracy: %g, Validation Accuracy:%g " %(i,train_accuracy, validation_accuracy)
                        if validation_accuracy >= bestAccuracy:
                            modelFile=self.modelFile+'_E'+str(e)+'_B'+str(i)
                            save_path = saver.save(sess, modelFile)
                            print "Model saved in file: ", save_path
                            
                            bestAccuracy = validation_accuracy
                            test_accuracy=self.computeAcuracy(self.testSet)
                        summary_str = summary_op.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                        summary_writer.add_summary(summary_str, i)
                print "Epoch: %d , best Validation Accuracy: %g, corresponding test Accuracy %g" %(e, bestAccuracy, test_accuracy)
                predictedLabel=tf.argmax(self.y_conv,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
                trueLabel= tf.argmax(self.y_,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
                fp.write("Time Stamp: %s, Test_Accuracy" %(test_accuracy))
                for i, (p,t) in enumerate(zip(predictedLabel,trueLabel)):
                    fp.write("Index: %d, Prediction: %d Truth:%d, exact Label:" %(i,p,t) , self.testSet[1][i] )

            sess.close()
            fp.close()
