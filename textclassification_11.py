"""
positive or negative sentiment classification
spam or not spam
text message or official doc
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Words are our features
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]

documents = []
# Category is neg or pos
for category in movie_reviews.categories():
    # fileid is name of file
    for fileid in movie_reviews.fileids(category):
        # append tuple ([all words in doc], neg/pos)
        documents.append((list(movie_reviews.words(fileid)), category))

# shuffle
random.shuffle(documents)
# print(documents)
# print(documents[1])

# get list of all words in all reviews
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# key and value, (word, freq)
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["stupid"])

# get list of first 3000 most common words, just words
word_features = list(all_words.keys())[:3000]


def find_features(document):
    """
    :param document: list of all words in the document
    :return: dictionary of words found in doc that were int he 3000 most common word list
    """

    # list to set gives unique elements
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# featuresets = [(find_features(rev), category) for (rev, category) in documents]

# feature sets will be a list of tuples
# tuple[0]: dictionary of words in review and if they were found in most common word list
# example tuple[0] = { the: True, hello: False }
# tuple[1]: category, neg or pos
# example featuresets[0] = ({ the: True, hello: False}, pos)

featuresets = []
for review, category in documents:
    featuresets.append((find_features(review), category))


training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive bayes algorithm ('stupid bayes')
# low proccessing demand

# posterior = (prior occurrences x likelihood )/ evidence ... likelihood of positive



# Pickle Load example
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()


# NLTK Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo Accuracy: ",
      (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(5)

# Pickle save example
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Naive Bayes Algo Accuracy: ",
      (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# Gaussian Niave Bayes
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier Naive Bayes Algo Accuracy: ",
#       (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

# Bernoulli Naive Bayes
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Naive Bayes Algo Accuracy: ",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC


# Logistic Regression Classfier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Accuracy: ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# Stochastic Gradient Descent
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_classifier Accuracy: ",
      (nltk.classify.accuracy(SGD_classifier, testing_set))*100)
# SGD_classifier.show_most_informative_features(5)

# Support Vector Classification
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier Accuracy: ",
      (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
# SVC_classifier.show_most_informative_features(5)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Accuracy: ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Accuracy: ",
      (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
