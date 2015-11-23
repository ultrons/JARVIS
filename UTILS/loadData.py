#!/usr/bin/python
"""
FileName: loadData.py
Abstract: Read Pre-Trained Word Vectors (Glove/Word2Vec) and
          And Corpus, and create a training, test and validation
          split.
Author  : Vaibhav Singh
Attribution:
------------
    -
Note:
-----
         Plenty of un-necessary libraries being imported in this module
         as of today (Need to do some clean up.
"""



#Standard Libraries
import cPickle, timeit
from pprint import pprint

#Third party libraries
import numpy as np
from sklearn.utils import shuffle


class preTrainedVectors(object):
    def __init__(self,vectorFiles):
        self.timeRecord={}
        self.wvList=[]
        for afile in vectorFiles:
            w2v={}
            print ("Reading File: %s" %afile)
            localStartTime=timeit.default_timer()
            fp=open(afile)
            for aline in fp:
                lineVector=aline.rstrip().split()
                word=lineVector[0]
                w2v[word]=lineVector[1:]
            self.wvList.append(w2v)
            self.timeRecord[afile]=timeit.default_timer() - localStartTime
            fp.close()

    def savePickle(self, pickleFile):
        self.word2vecPickle=pickleFile
        pickleDumpStart=timeit.default_timer()
        print ("Dumping Pickle: %s" %self.word2vecPickle)
        fp=open(word2vecPickle, 'wb')
        cPickle.dump(self.wvList, fp)
        fp.close()
        self.timeRecord['pickleDump']=timeit.default_timer()-pickleDumpStart

class corpus(object):
    #Vector parameters
    # dataSet Format:
    #    sst: (phraseFile, labelFile)
    #    pol:  
    #    xxx:  
    def __init__(self,preTrainedVectors, dataSet,format='sst',maxwords=60,wvDim=100):
        #Load Pretrained vectors
        #super(preTrainedVectors,self).__init__()
        # Parse Corpus
        self.wvList=preTrainedVectors.wvList
        self.wvDim=wvDim
        self.maxwords=maxwords
        dX=[]
        dY=[]
        if format == 'sst': # Stanford Tree Bank format
            phraseDictionaryFile, labelFile = dataSet
            # Reading phrase dictionary
            with open(phraseDictionaryFile, 'r') as f:
                print ("Format: %s, Reading %s ....."  %(format,
                                                         phraseDictionaryFile))
                vectorDict={}
    
    
                for aline in f:
                    phrase,id=aline.rstrip().split('|')
                    vector=[]
                    #Prepare Word vectors
                    for count, w in enumerate(phrase.split()):
                        vector.append(self.word2Vector(w))
                    # Zero padding for smaller sentence/phrase
                    for i in range(count+1,self.maxwords):
                        vector.append(['0']*self.wvDim)
                    vectorDict[id]=vector
            f.close()
            # Reading Label dictionary
            with open(labelFile, 'r') as f:
                print ("Format: %s, Reading %s ....."  %(format, labelFile))
                for aline in f:
                    id,p=aline.rstrip().split('|')
                    # Just to work with mismatched subsets in trials
                    if id in vectorDict:
                        p=float(p)
                        if   p <= 0.2: label=[1,0,0,0,0]
                        elif p <= 0.4: label=[0,1,0,0,0]
                        elif p <= 0.6: label=[0,0,1,0,0]
                        elif p <= 0.8: label=[0,0,0,1,0]
                        elif p <= 1.0: label=[0,0,0,0,1]
                        dX+=vectorDict[id]
                        dY.append(label)
            f.close()
            self.numExamples=len(dY)
            #print dX
            self.X=np.array(dX,
                       dtype='float64').reshape((self.numExamples,self.maxwords*self.wvDim))
            #self.Y=np.array(dY, dtype='int')
            self.Y=np.array(dY, dtype='int').reshape((self.numExamples,5))

        if format == 'mr': #https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
            # rt-polaritydata is organizd as:
            # rt-polaritydata/rt-polarity.neg, rt-polaritydata/rt-polarity.pos
            # in this case the dataset is expected as neg, pos file
            negSet, posSet = dataSet
            with open(negSet, 'r') as f:
                print ("Format: %s, Reading %s ....."  %(format,negSet))
                for aline in f:
                    for count, w in enumerate(aline.rstrip().split()):
                        dX.append(self.word2Vector(w))
                    # Zero padding for smaller sentence/phrase
                    for i in range(count+1,self.maxwords):
                        dX.append(['0']*self.wvDim)
                    dY.append([1,0])
            f.close()
            with open(negSet, 'r') as f:
                print ("Format: %s, Reading %s ....."  %(format,posSet))
                for aline in f:
                    for count, w in enumerate(aline.rstrip().split()):
                        dX.append(self.word2Vector(w))
                    # Zero padding for smaller sentence/phrase
                    for i in range(count+1,self.maxwords):
                        dX.append(['0']*self.wvDim)
                    dY.append([0,1])
            f.close()
            self.numExamples=len(dY)
            self.X=np.array(dX,
                       dtype='float64').reshape((self.numExamples,self.maxwords*self.wvDim))
            #self.Y=np.array(dY, dtype='int')
            self.Y=np.array(dY, dtype='int').reshape((self.numExamples,2))
    def word2Vector(self,word):
        # set default vector to 0
        vector=['0']*self.wvDim
        for wvd in self.wvList:
            if word in wvd: return wvd[word]
        return vector

    def createSplit(self, ttvSplit=[0.80,0.10,0.10]):
        tr,tst,vld = ttvSplit
        self.X, self.Y = shuffle(self.X, self.Y, random_state=97)
        N=self.numExamples
        train=int(N*tr)
        test=train+int(N*tst)
        return (self.X[:train],self.Y[:train]), \
                (self.X[train:test],self.Y[train:test]),  \
                (self.X[test:],self.Y[test:])
        
if __name__ == '__main__':
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
               
    pv=preTrainedVectors(preTrainedVecFiles)
    format='mr'
    dataSet=corpus(pv,DS[format],format)
    d1,d2,d3=dataSet.createSplit()
    print type(d3[1]), d3[1].shape, type(d3[0]), d3[0].shape
    



