# Sentiment-Analysis-of-Tweets-

Computing sentiments of Twitter data to get insight into 2019 Canadian elections. 

Central to sentiment analysis are techniques first developed in text mining. Some of those techniques require a large collection of classified text data often divided into two types of data, a training data set and a testing data set. The training data set is further divided into data used solely for the purpose of building the model and data used for validating the model. The process of building a model is iterative, with the model being successively refined until an acceptable performance is achieved. The model is then used on the testing data in order to calculate its performance characteristics.

In this project, two sets of data are used. 
1. generic_tweets.txt file contains tweets that have had their sentiments already analyzed and recorded as binary values 0 (negative) and 4
(positive).

2. Canadian_elections_2019.csv, contains a list of tweets regarding the 2019 Canadian elections.

Chronology followed:
1. Data Cleaning
2. Exploratory data analysis: determine political party of given tweet (using) and word clouds
3. Model preparation: multiple classification algorithms for generic tweets (logistic regression, k-NN, Naive Bayes, SVM, decision trees, ensembles (RF, XGBoost)), where each tweet is considered a single observation/example. 
4. Model Implementation: Implementing models on test data and best performing model used to predict sentiment of Canadian election tweets
