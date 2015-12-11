"""
FileName: convNet_tensorFlow_yck_custom.py
Abstract: Implementation of CNN proposed by Yoon Kim:
    http://arxiv.org/pdf/1408.5882v2.pdf
Author  : Vaibhav Singh


Attribution:
------------
"""
import tensorflow as tf
from sklearn.utils import shuffle
import loadData as ld
from datetime import datetime

class CNN_YCK(object):
    # IMPROVE: As an after thought training/optimization parameters are better with train
    # sub routine
    def __init__(self,mini_batch_size,epochs, optimizer,
                 trainSet, testSet, valSet, modelFile, startPoint, normMax
                 ):
        tf.set_random_seed(786)
        self.mini_batch_size=mini_batch_size
        self.epochs=epochs
        self.trainSet=trainSet
        self.testSet=testSet
        self.valSet=valSet
        self.optimizer=optimizer
        self.numClasses=trainSet[1].shape[1]
        self.trainPointer=0
        self.trainSetSize=trainSet[1].shape[0]
        #Model Name
        modelFileSuffix=str(datetime.now())
        modelFileSuffix=modelFileSuffix.replace(' ','_')
        modelFileSuffix=modelFileSuffix.replace(':','_')
        self.modelFile=modelFile+'_'+modelFileSuffix+'.YCK'
        self.startPoint=startPoint
        self.normMax=normMax

    def setInput(self, imageX, imageY,indexTable=None, word2vecShape=None):
        self.imageX=imageX
        self.imageY=imageY
        if indexTable is not None:
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("embedding", word2vecShape)
            self.x = tf.placeholder(tf.int32, shape=[None, imageX])
            self.x_image = tf.nn.embedding_lookup(self.embedding, self.x)
            self.x_image = tf.reshape(self.x_image,[-1, imageX,imageY, 1], name="Input_Image")
            self.y_ = tf.placeholder('float', shape=[None, self.numClasses],
                                     name="Input_Labels")
        else:
            with tf.name_scope("INPUTS"):
                self.x = tf.placeholder('float', shape=[None, imageX*imageY],
                                    name="1D_input_vector" )
                self.x_image = tf.reshape(self.x,[-1, imageX,imageY, 1], name="2D_input_image")
                self.y_ = tf.placeholder('float', shape=[None, self.numClasses],
                                     name="Input_Labels")

        
    # Shape is [h,k] where h is size of n-gram
    # k is word vector dimension
    # Shape of W has form [imageX, imageY, nChannnels, nfeature_maps]
    def weight_variable(self,shape, id):
        return tf.get_variable("weights_"+id, shape,
                               initializer=tf.random_normal_initializer(mean=0,
                                                                        stddev=0.001, seed=786))
    def bias_variable(self,shape, id):
        return tf.get_variable("bias/"+id, shape, initializer=tf.constant_initializer(0.1))

    # no zero padding,  aparent from
    # Eq. (3) c = [c1,c2,...,cn-h+1],
    # Hence padding='VALID' has been used
    def conv2d(self,x,W,strideX=1, strideY=1,padding='VALID'):
        return tf.nn.conv2d(x,W, strides=[1,strideX,strideY,1], padding=padding,
                           name="CONVOLVE")
    # Kim has used,  Max over time pooling (Collobert et al., 2011)
    def max_pool(self,x, poolX, poolY,padding, pool_id):
        return tf.nn.max_pool(x, ksize=[1,poolX,poolY,1], strides=[1,poolX,poolY,1],
                              padding='SAME')
    
    def build(self):
        pooledSamples=0
        h_conv=[]
        h_pool=[]
        W_conv=[]
        b_conv=[]
        n_featureMaps=100
        k=self.imageY
        for h in [3, 4, 5]:
            with tf.name_scope("CONV1"):
                with tf.name_scope("%d_gram" %h):
                    W_conv.append(self.weight_variable([h, k, 1,
                                                        n_featureMaps], "%d_gram" %h))
                    b_conv.append(self.bias_variable([n_featureMaps], "%d_gram" %h))
                h_conv.append(tf.nn.relu(self.conv2d(self.x_image, W_conv[-1]) + b_conv[-1]))
            with tf.name_scope("POOL1"):
                h_pool.append(self.max_pool(h_conv[-1],self.imageX-h+1,self.imageY-k+1,'%d_gram_maxOverTime',h))
            pooledSamples+=n_featureMaps
                               
        with tf.name_scope("DROPOUT"):
            h_pool_r=[tf.reshape(x, [-1]) for x in h_pool]
            h_pool_flat = tf.reshape(tf.concat(0, h_pool_r), [-1, pooledSamples])
            self.keep_prob = tf.placeholder('float', name="keep_prob")
            h_fc1_drop=tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("SOFTMAX"):
            # Finally Weights and biases for output softmax layer
            W_fc2 = self.weight_variable([pooledSamples, self.numClasses],
                                         "softmax")
            self.W=W_conv+[W_fc2]
            b_fc2 = self.bias_variable([self.numClasses], "softmax")
            self.b=b_conv+[b_fc2]
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
       # print b[0] 
       # exit()
        return b,self.trainPointer

    def computeAcuracy(self, data):
        Pointer=0
        #batch_size=data[1].shape[0]
        batch_size=500
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
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv)) 
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
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

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
                if embeddingMatrix is not None:
                    sess.run(self.embedding.assign(embeddingMatrix))
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
                                              self.keep_prob: 0.5}) #DROP
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
                print "End of Epoch: %d , best Validation Accuracy till date: %g, corresponding test Accuracy %g" %(e, bestAccuracy, test_accuracy)
                predictedLabel=tf.argmax(self.y_conv,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
                trueLabel= tf.argmax(self.y_,1).eval(feed_dict={self.x:self.testSet[0], self.y_: self.testSet[1], self.keep_prob: 1.0})
                fp.write("Time Stamp: %s, Test_Accuracy" %(test_accuracy))
                for i, (p,t) in enumerate(zip(predictedLabel,trueLabel)):
                    fp.write("Index: %d, Prediction: %d Truth:%d, exact Label:" %(i,p,t) , self.testSet[1][i] )
            sess.close()      
            fp.close()
