#Semantics Based Sentence Classification
########################################
ProblemStatement:
------------------
 Performanobjectivecomparisonoftwoprominentmethodsincontextof Semantic Based Sentence Classification: [2] “Tree structured Long Short Term Memory Networks” and [3] “Convolutional Network for Sentence Classification”. We also plan to extend multi channel design (static and task based vector usage) utilized in the later work [3], to explore Compound Multi­Channel Neural Networks (parallely trained vectors from two different neural architectures).
Input/Output Behaviour:
-----------------------
Stanford Sentiment Treebank classifies sentiments in 5 classes using the following cut­offs:
(0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0], for very negative, negative, neutral, positive, very positive, respectively. **
￼Sentence
1. the worst movie of 2002
2. (Allen's) been making piffle for a long while
3. Walt Becker 's film pushes all the
demographically appropriate comic buttons
Sentiment Sentiment Label Class 0.013889 1
0.30556 2
0.80556 5
Evaluation Metrics:
-------------------
 Test set accuracies in Stanford Sentiment Treebank [1]
Data Set:
---------
 We plan to use use Stanford Sentiment Treebank [1]. There are two subtasks:
1. Binary classification of sentences. We plan to use the standard train/dev/test splits of
6920/872/1821
2. Fine grained classification over five classes: very negative, negative, neutral, positive,
and very positive. We plan to use the standard train/dev/test splits of 8544/1101/2210
Baseline:
 Binary classification: 82.4% test set accuracy [2] Fine grained classification: 43.2% test set accuracy [2]
Oracle:
-------
 By definition of oracle, test set accuracy should be 100% (human level comprehension). So far, the best test set accuracy achieved by any model is [2] “Tree structured Long Short Term Memory Networks” which is 86.9% on binary classification and 50.6% on fine grained classification.
Challenges: Sentence classification is an inherently difficult problem to solve because there are not perfect model to grasp human level comprehension of language. Traditional methods like bag of words fail to generalize because meaning of a sentence is order sensitive (e.g., “cats climb trees” vs. “trees climb cats”). Order sensitive models on the other hand have limited scalability. Larger window of the phrase makes the model harder to train using conventional methods. Fully connected neural architecture is hard to train and require a very large data set
due to inherently large number of parameters. Traditional RNN suffer from vanishing­exploding gradient problems. Recursive neural networks require parse tree structure information. Convolutional neural network and Long Short Term tree based RNN methods which we explore in depth, address some of these issues. Yet accuracy reported by these works clearly indicates a scope of further improvement. In the current work we go into depth of these issues and present the essence of techniques, deriving insights and if possible try to improve the results.
Related Work:
--------------
[1] Socher, Richard, Alex Perelygin, Jean Y Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In ​Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).
[2] Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations from tree­structured long short­term memory networks. ACL, 2015.
[3] Yun Kim, Convolutional Neural Networks for Sentence Classification. EMNLP,2014.
[4] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. ​Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
[5] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. ​Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013
** Described in Stanford Sentiment Treebank Dataset Documentation. (http://nlp.stanford.edu/sentiment/index.html)
