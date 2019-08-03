# machineLearning_sentimentAnalysis
project2 for comp90049 knowledge technologies
The main goal for this assignment is to identify people's sentiments by analysing their tweets.
## Datasets
Three kind of datasets are given for different purposes. The **‘train’** related tweets data sets are leveraged to train classifier models with various machine learning algorithms while the **‘eval’** related data sets are used to evaluate the performance of developed models. 
## Scikit-Learn
**Scikit-Learn** is leveraged in this project as a tool to build sentiment analysis models and predict for the unseen dataset. It is a popular machine learning library designed for Python which provides a robust set of algorithms including Naïve Bayes, Decision Trees and so on. The main process is:
1) At first, extract and select feature words from tweets after tokenizing, counting and normalizing.  
2) Secondly, train the classifier model with given the train data set utilizing build-in method based on certain machine learning algorithm.  
3) After building the model, evaluate it with given evaluation data sets and calculate evaluation metrics with output results.  
4) Last step, the well trained classifier model allots labels for given unlabeled testing data.  
## Feature Engineering
To improve the performance of analysis model, feature engineering has to be conducted. In this assignment, the feature selection tool provided by **Scikit-Learn** is used to gather more feature words to identify emotions. The main steps:
1) Tokenizing. Convert each tweet text into a sequence of tokens.  
2) Data cleaning. Remove all punctuations, and stop words. Stop words refers to commonly used words such as “a”, “the”, “in” which do no help to train a model.  
3) Featuring selection. After feature extraction, we may obtain a huge number unigram feature words. Only top 5000 most frequently occurring words are concerned in this project.  
4) Train the classifier with selected features.  
