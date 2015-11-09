#!/usr/bin/python
#Standard Libraries
import cPickle
import gzip
from pprint import pprint

#Third party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


class loadData(object):
    #Vector parameters
    MAXWORDS=10
    SWORDVEC=300
    SPLIT=[0.40,0.40,0.20]
    def __init__(self,wordEmbdfile,format='txt', dataSet=None,labels=None):
        #read word embeddings (word2vec/glove/...)
        if format == 'txt':
            #wordEmbd=np.genfromtxt(wordEmbdfile, dtype=str)
            fp=open(wordEmbdfile)
            self.wordVectors={}
            # Create wordVector dictionary
            for aline in fp:
                w=aline.rstrip().split()
                self.wordVectors[w[0]]=w[1:]
            fp.close()

            #Save as pickle
            with open("wordVectors.p", 'wb') as f: 
                cPickle.dump(self.wordVectors,f )
        elif format == 'pickle':
            with open(wordEmbdfile, 'rb') as f: 
                self.wordVectors=cPickle.load(f)
        else:
            print("Format: %s? You gotta be kidding me :)" %format)
            exit()

        #Assuming dataset in sst format (phrase|phraseid) and labels
        # are in format id|labels
        f=open(dataSet)
        self.data={}
        for aline in f:
            phrase,id=aline.rstrip().split('|')
            id=int(id)
            dataVector=[]
            #Prepare Word vectors
            x=None
            zeroVector=np.zeros((self.SWORDVEC,1), dtype='float32')
            for count, w in enumerate(phrase.split()):
                if w in self.wordVectors:
                    vector=np.matrix(self.wordVectors[w], dtype=float32).reshape((self.SWORDVEC,1))
                else:
                    vector=zeroVector
                if x is None:
                    x=vector
                else:
                    x=np.vstack((x,vector))
            # Zero padding for smaller sentence/phrase
            for i in range(count+1,self.MAXWORDS):
                x=np.vstack((x,zeroVector))
            #dataVector.append(x)
            #self.data.append(dataVector)
            self.data[id]=x
        f.close()

        f=open(labels)
        self.dataSet={}
        for aline in f:
            id,p=aline.rstrip().split('|')
            p=float(p)
            id=int(id)
            if p <=0.2:
                label=-2
            elif p <= 0.4:
                label=-1
            elif p <= 0.6:
                label=0
            elif p <= 0.8:
                label=1
            elif p <= 1.0:
                label=2
            #self.data[id].append(label)
            #self.data[id]=np.array(self.data[id])
            self.data[id]=np.append(self.data[id], label)
        f.close()


    def createSplit(self):
        dataShuffled=np.vstack(self.data.values())
        l=len(self.data.values())
        np.random.shuffle(dataShuffled)
        corpusSize=len(dataShuffled)
        train=int(corpusSize*self.SPLIT[0])
        test=train+int(corpusSize*self.SPLIT[1])
        return dataShuffled[:train], dataShuffled[train:test], dataShuffled[test:]
        



            



        




if __name__ == '__main__':
    #glove='/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/glove'
    #dataSet='/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/sst/dictionary.txt'
    #labels='/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/sst/sentiment_labels.txt' 
    glove='/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/sample'
    wordVectorPickle='/Users/MAVERICK/Documents/CS221/project/work_area/JARVIS/wordVectors.p'
    dataSet='/Users/MAVERICK/Documents/CS221/project/work_area/JARVIS/sampleDictionary.txt'
    labels='/Users/MAVERICK/Documents/CS221/project/work_area/JARVIS/sampleLables.txt'
    #dataSet=loadData(wordVectorPickle, 'pickle', dataSet, labels)
    dataSet=loadData(glove, 'txt', dataSet, labels)
    pprint(dataSet.data)
    



