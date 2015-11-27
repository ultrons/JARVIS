from distutils.core import setup
setup(
      name='JARVIS',
      version='1.0',
      package_dir={'loadData': 'UTILS',
                   'convNet_tensorFlow': 'CONVNET',
                   'convNet_theano': 'CONVNET'
                    },
      packages=['loadData','convNet_tensorFlow','convNet_theano'  ]
      )
