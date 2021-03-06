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
from convNet_tensorFlow import *
import convNet_tensorFlow_yck_custom as YCK
#from convNet_theano import *

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
                    help=default('Pre-Trained VECTOR to Use'),
                    metavar='VECTOR', default='word2vec')
  parser.add_option('-c', '--corpus', dest='corpus',
                    help=default('Corpus to Use'),
                    metavar='CORPUS', default='mr-toy')
  parser.add_option('-l', '--method', dest='library',
                    help=default('LIBRARY to Use'),
                    metavar='LIBRARY', default='tensorFlow')
  parser.add_option('-m', '--modelFile', dest='modelFile',
                    help=default('MODELFILE , model will be saved to'),
                    metavar='MODELFILE', default='./model.f')
  parser.add_option('-s', '--startPoint', dest='startPoint',
                    help=default('STARTPOINT , MODELFILE to load'),
                    metavar='MODELFILE', default=None)
  parser.add_option('-g', '--ctype', dest='classificationType',
                    help=default('TYPE , classification Choice'),
                    metavar='TYPE', default='fineGrained')
  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0:
    raise Exception('Command line input not understood: ' + str(otherjunk))
  args = dict()
  args['info']=options.info
  args['corpus']=options.corpus
  args['preTrainVec']=options.preTrainVec
  args['trainConfig']=options.trainConfig
  args['modelFile']=options.modelFile
  args['networkConfig']=options.networkConfig
  args['library']=options.library
  args['startPoint']=options.startPoint
  args['classificationType']=options.classificationType
  return args




def jarvis (info, corpus, preTrainVec, networkConfig, trainConfig, library,
            modelFile, startPoint, classificationType):

    # Available Pre-Trained Vectors
    preTrainedVecFiles={ 'glove':
                    '/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/glove/glove.840B.300d.txt' ,
                     'word2vec':
                    '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/vectors.6B.100d.txt',
                     'word2vec-toy':
                    '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/vectors.6B.100d.splitted.aaa'
                       }
    # Available Corpuses
    dataSetFiles = { 'sst-toy':
                # (phraseFile,labelFile) 
                ('/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleDictionary.txt',
                 '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/sampleLables.txt'
                ),
           'mr-toy':
               #(Neg,Pos)
               ('/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/Sample.rt-polarity.neg',
                '/Users/MAVERICK/Documents/CS221/project/work_area/SCRATCH/Sample.rt-polarity.pos'
               ),

           'mr':
               #(Neg,Pos)
               ('/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.neg',
                '/Users/MAVERICK/Documents/CS221/project/work_area/MR/rt-polaritydata/rt-polarity.pos'
               ),
            'sst': 
                    ('/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/sst/dictionary.txt',
                     '/Users/MAVERICK/Documents/CS221/project/work_area/treelstm/data/sst/sentiment_labels.txt'
                    ),
            'sst2': 
                    (
                        '/Volumes/Veronica/PROJECTS/WORK_AREA/sentlabels_train.txt',
                        '/Volumes/Veronica/PROJECTS/WORK_AREA/sentlabels_test.txt',
                        '/Volumes/Veronica/PROJECTS/WORK_AREA/sentlabels_val.txt'
                    )
              }
    # Available Network Options
    networkConfigSet = {
        'CNN-0': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(5, 5)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-1': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(3, 50), (4, 50), (5, 50)], 20, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-2': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(3, 50), (4, 50), (5, 50)], 20, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-3': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(4, 4), (3, 3)], 32, 2, 2, 1, 1),
                ([(2, 2), (5, 5)], 32, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':100,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-4': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(4, 4), (3, 3)], 32, 2, 2, 1, 1),
                ([(2, 2), (5, 5)], 32, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-5': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(4, 4), (3, 3)], 32, 2, 2, 1, 1),
                ([(2, 2), (5, 5)], 32, 2, 2, 1, 1),
                ([(6, 6), (8, 8)], 32, 5, 5, 1, 1)
            ],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-6': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(4, 4), (3, 3)], 32, 2, 2, 1, 1),
                ([(2, 2), (5, 5)], 32, 2, 2, 1, 1),
                ([(6, 6), (8, 8)], 32, 5, 5, 1, 1)
            ],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },

        'CNN-7': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(3, 100), (4, 100), (5, 100)], 100, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-8': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(2, 2), (2, 2), (4, 4), (4,  4) ],    16, 2, 2, 1, 1),
                ([(5, 5), (7, 7), (9, 9), (11, 11)],    16, 2, 2, 1, 1)
            ],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
          'CNN-9': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(5, 5), (7, 7)], 20, 2, 2, 1, 1)],
            #'convPoolLayers':[([(3, 100), (4, 100)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-10': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(5, 5), (3,3)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-11': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(5, 5)],    20, 2, 2, 1, 1),
                ([(7, 7)],    20, 2, 2, 1, 1)
            ],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-12': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(5, 5)],    20, 2, 2, 1, 1),
                ([(7, 7)],    20, 2, 2, 1, 1),
                ([(11, 11)],    20, 5, 5, 1, 1)
            ],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-13': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(7, 10)],    10, 2, 2, 1, 1),
                ([(5, 10)],    10, 2, 2, 1, 1)
            ],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':32,
            # SoftMaxLayer
            'softMaxLayerDim':2
        },
        'CNN-14': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[
                ([(7, 10)],    10, 2, 2, 1, 1),
                ([(5, 10)],    10, 2, 2, 1, 1)
            ],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':32,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },
        'CNN-15': {
            'type': 'CNN',
            # Max words per phrase
            # Also decides X Dimension of the phrase image
            # Y dimension comes from size of word vectors
            'maxWords': 60,
            # (FilterX, FilterY, filterCount, poolX, poolY, strideX,strideY )
            'convPoolLayers':[([(5, 10), (3,10), (2,10)], 20, 2, 2, 1, 1)],
            # Assuming there is only one fully connected layer
            'fullyConnectedLayerDim':60,
            # SoftMaxLayer
            'softMaxLayerDim':5
        },




    }
    # Available Training Options
    trainConfigSet = {
        'T-toy': {
            'mini_batch_size':50,
            'epochs':20,
            'optimizer': 'ADAM',
            'keep_prob': 0.5,
            'dataSplit': [0.9,0.05,0.05],
            'normLimit': 3
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
        if library != 'custom_tune':
            # Load dataSet
            dataSet=ld.loadCorpus(pv,dataSetFiles,corpus,wvDim=pv.wvDim, maxwords=networkConfigSet[networkConfig]['maxWords'])
        # Network Specs
        networkSpec=networkConfigSet[networkConfig]
        fullyConnectedLayerDim=networkSpec['fullyConnectedLayerDim']
        softMaxLayerDim=networkSpec['softMaxLayerDim']
        # TrainingSpecs
        trainingSpec=trainConfigSet[trainConfig]
        mini_batch_size=trainingSpec['mini_batch_size']
        epochs=trainingSpec['epochs']
        optimizer=trainingSpec['optimizer']
        dataSplit=trainingSpec['dataSplit']
        # Construct Network
        if networkSpec['type'] == 'CNN':
            imageX=networkSpec['maxWords']
            imageY=pv.wvDim
            if  library == 'theano': 
                training_data, test_data, validation_data = dataSet.createSplit(dataSplit)
                layers=[]
                channels=1
                for (filterSpec, poolX,poolY, strideX, strideY) in self.networkSpec['convPoolLayers']:
                    for (filterX,filterY,filterCount) in filterSpec:
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
            if  library == 'tensorFlow':
                training_data, test_data, validation_data = dataSet.createSplit(dataSplit)
                network=tfNetwork(mini_batch_size, epochs, optimizer,
                                  networkSpec, training_data,
                                 test_data, validation_data, modelFile,
                                  startPoint, trainingSpec['normLimit'])
                network.setInput(imageX, imageY)
                network.build()
                network.train(ctype=classificationType)
                network.printEval()
            if  library == 'custom':
                training_data, test_data, validation_data = dataSet.createSplit(dataSplit)
                network=YCK.CNN_YCK(mini_batch_size, epochs, optimizer,
                                   training_data,
                                 test_data, validation_data, modelFile,
                                  startPoint, trainingSpec['normLimit'])
                network.setInput(imageX, imageY)
                network.build()
                network.train(ctype=classificationType)
                network.printEval()
            if  library == 'custom_tune':
                indexTable, word2vec_matrix = pv.genIndexTable()
                dataSet=ld.loadCorpus(pv,dataSetFiles,corpus,wvDim=pv.wvDim,
                                      maxwords=networkConfigSet[networkConfig]['maxWords'],
                                      indexTable=indexTable
                                     )
                training_data, test_data, validation_data = dataSet.createSplit(dataSplit)
                network=YCK.CNN_YCK(mini_batch_size, epochs, optimizer,
                                   training_data,
                                 test_data, validation_data, modelFile,
                                  startPoint,
                                    trainingSpec['normLimit'] )
                network.setInput(imageX, imageY, indexTable=indexTable,
                                 word2vecShape=word2vec_matrix.shape)
                network.build()
                network.train(embeddingMatrix=word2vec_matrix,ctype=classificationType)
                network.printEval()





            

if __name__ == '__main__':
  args = readCommand( sys.argv[1:] )
  jarvis( **args )
  pass

