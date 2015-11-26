"""
FileName: jarvis.py
Abstract: Basic framework to launch learning through one of the supported
          techniques:
              - CONVNET (using theano library)
              - CONVNET (using tensor flow library)
              - RNN     (using torch/lua flow library)
Author  : Vaibhav Singh


Attribution:
------------
    - See source of respective subcomponents for attribution
"""
import os, sys
from pprint import pprint

import loadData as ld
#from convNet_tensorFlow import *
from convNet_theano import *

def default(str):
  return str + ' [Default: %default]'
  
def readCommand( argv ):
  """
  Processes the command used to run the training
  """
  from optparse import OptionParser
  usageStr = """
  USAGE:      python jarvis.py <options>
  EXAMPLES:   (1) python jarvis.py
                  - Run Default Training on Default Architecture with Default
                  Settings :-)
  """
  parser = OptionParser(usageStr)

  parser.add_option('-i', '--info', action="store_true",
                    help=default('Print available Configs for Network INFO and Training'),
                    metavar='INFO', default=False)
  parser.add_option('-n', '--networkConfig', dest='networkConfig', type='str',
                    help=default('Neural Network Configuration'),
                    metavar='NETWORKCONFIG',
                    default='CNN-1')
  parser.add_option('-t', '--trainConfig', dest='trainConfig',
                    help=default('Training Configrations'),
                    metavar='TRAINCONFIG', default='T-toy')
  parser.add_option('-p', '--preTrainVec', dest='preTrainVec',
                    help=default('Pre-Trained Vector to Use'),
                    metavar='TRAINCONFIG', default='word2vec')
  parser.add_option('-c', '--corpus', dest='corpus',
                    help=default('Corpus to Use'),
                    metavar='CORPUS', default='sst-toy')
  parser.add_option('-l', '--method', dest='library',
                    help=default('LIBRARY to Use'),
                    metavar='LIBRARY', default='theano')
  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0:
    raise Exception('Command line input not understood: ' + str(otherjunk))
  args = dict()
  args['info']=options.info
  args['corpus']=options.corpus
  args['preTrainVec']=options.preTrainVec
  args['trainConfig']=options.trainConfig
  args['networkConfig']=options.networkConfig
  args['library']=options.library
  return args




def jarvis (info, corpus, preTrainVec, networkConfig, trainConfig, library):
    # Available Pre-Trained Vectors
    preTrainedVecFiles={ 'glove':
                    '/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/glove/glove.840B.300d.txt' ,
                     'word2vec':
                    '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/vectors.6B.100d.splitted.aaa'
                       }
    # Available Corpuses
    dataSetFiles = { 'sst-toy':
                # (phraseFile,labelFile) 
                ('/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleDictionary.txt',
                 '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleLables.txt'),
           'mr':
               #(Neg,Pos)
               ('/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.neg',
                '/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.pos'
               )
              }
    # Available Network Options
    networkConfigSet = {
        'CNN-1': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[(5, 5, 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':100,
            # SoftMaxLayer
            'softMaxLayerDim':5
        }
    }
    # Available Training Options
    trainConfigSet = {
        'T-toy': {
            'mini_batch_size':10,
            'epochs':3,
            'optimizer': 'ADAGRAD',
            'dropOutProb': 0.5,
            'dataSplit': [0.6,0.3,0.1]
        }
    }
    # Print available Configs
    if info :
        pprint ("Available Networks:")
        pprint (networkConfigSet)
        pprint ("")
        pprint ("Available Training Options:")
        pprint (trainConfigSet)
        pprint ("")
        pprint ("Available Pre-Trained Vectors:")
        pprint (preTrainedVecFiles)
        pprint ("")
        pprint ("Available Corpus:")
        pprint (corpus)
        pprint ("")
        pprint ("Carpe Diem !!!")
        exit()


    else:    
        preTrVecFiles=[preTrainedVecFiles[preTrainVec]]
        # Load Pre-Trained Vectors
        pv=ld.preTrainedVectors(preTrVecFiles)
        # Load dataSet
        dataSet=ld.loadCorpus(pv,dataSetFiles,corpus,wvDim=pv.wvDim)
        # Network Specs
        network=networkConfigSet[networkConfig]
        fullyConnectedLayerDim=network['fullyConnectedLayerDim']
        softMaxLayerDim=network['softMaxLayerDim']
        # TrainingSpecs
        trainingSpec=trainConfigSet[trainConfig]
        mini_batch_size=trainingSpec['mini_batch_size']
        epochs=trainingSpec['epochs']
        dataSplit=trainingSpec['dataSplit']
        # Construct Network
        if network['type'] == 'CNN':
            training_data, validation_data, test_data = load_data_shared(dataSet.createSplit(dataSplit))
            imageX=network['maxWords']
            imageY=pv.wvDim
            if  library == 'theano': 
                layers=[]
                channels=1
                for (filterX,filterY,filterCount,poolX,poolY, strideX, strideY) in network['convPoolLayers']:
                    layers.append(
                        ConvPoolLayer(image_shape=(mini_batch_size, channels, imageX, imageY), 
                        filter_shape=(filterCount, channels, filterX, filterY), 
                        poolsize=(poolX, poolY),
                        stride=(strideX,strideY),
                        border_mode='valid')
                    )
                    # Number of channels for next layer would be number of
                    # filters in the current layer
                    channels=filterCount
                    imageX=int((imageX-filterX)/strideX)+1
                    imageY=int((imageY-filterY)/strideY)+1
                # Add fully connected layer
                layers.append(
                FullyConnectedLayer(n_in=int(channels*imageX*imageY/(poolX*poolY)),
                                    n_out=fullyConnectedLayerDim))
                layers.append(
                SoftmaxLayer(n_in=fullyConnectedLayerDim, n_out=softMaxLayerDim))
                net = CNN_TH(layers, mini_batch_size)
                net.SGD(training_data, epochs, mini_batch_size, 0.5,
                        validation_data, test_data)
            

if __name__ == '__main__':
  args = readCommand( sys.argv[1:] ) # Get game components based on inputa
  jarvis( **args )
  # import cProfile
  # cProfile.run("runGames( **args )")
  pass

