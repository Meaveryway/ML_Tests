print ("Warining: Needs to be run on python3 because of MLxtend")

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

dataset = pd.read_csv('glass.csv')



print ("\n\nTraining a Naive Bayes classifier")
## Naive Bayes Classifier
features_list = dataset.columns
features = dataset.drop(columns=['Type'])
labels = dataset['Type']

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
print ("Score for Naive Bayes =", clf1.score(features_test, labels_test))
scores1 = cross_val_score(clf1, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print ("Mean score with Stratified 3x10 CV:", np.mean(scores1))

from sklearn.ensemble import AdaBoostClassifier
clf2 = AdaBoostClassifier(n_estimators = 10)
clf2.fit(features_train, labels_train)
print ("Score for AdaBoost = ", clf2.score(features_test, labels_test))
scores2 = cross_val_score(clf2, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print ("Mean score with Stratified 3x10 CV:", np.mean(scores2))

from mlxtend.evaluate import paired_ttest_5x2cv
t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf2, X=features, y=labels)
print("p-value = ", p)
if p <= 0.05:
	print('Difference between mean performance is probably real')
else:
	print('Algorithms probably have the same performance')
