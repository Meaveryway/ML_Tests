import pandas as pd 
import seaborn as sn
import matplotlib.pyplot as plt 

dataset = pd.read_csv('glass.csv')

print "\n First 5 rows:"
print dataset.head()
print "\n Last 5 rows:"
print dataset.tail()

print "\n Informations about the table:"
print dataset.info()

print "\n Table description:"
print dataset.describe()

print "\n Checking for null values:"
print dataset.isnull().sum()

print "\n Checking for feature correlation:"
corr_matrix = dataset.corr()
sn.heatmap(corr_matrix, annot=True)

#plt.show()

plot = plt.figure(figsize = (20, 4))

# Viewing distribution of labels:
plot.add_subplot(1,3,1)
sn.countplot(dataset['Type'])

# Box & Whishker plot for Na:
plot.add_subplot(1,3,2)
sn.boxplot(dataset['Na'])

# Distplot for Mg:
plot.add_subplot(1,3,3)
sn.distplot(dataset['Mg'])

#plt.show()
print "\n\nTraining a Naive Bayes classifier"
## Naive Bayes Classifier
features_list = dataset.columns
features = dataset.drop(columns=['Type'])
labels = dataset['Type']

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
print "Score before scaling =", clf.score(features_test, labels_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standard_scaled_features = scaler.fit_transform(features)
features_train, features_test, labels_train, labels_test = train_test_split(standard_scaled_features, labels, test_size = 0.2, random_state = 42)
clf.fit(features_train, labels_train)
print "Score after standard scaling =", clf.score(features_test, labels_test)

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
standard_scaled_features = scaler2.fit_transform(features)
features_train, features_test, labels_train, labels_test = train_test_split(standard_scaled_features, labels, test_size = 0.2, random_state = 42)
clf.fit(features_train, labels_train)
print "Score after min-max scaling= ", clf.score(features_test, labels_test)
print "\n we can conclude that Naive Bayes classifiers are invariant to scaling (they perform it internally)"

from sklearn.feature_selection import SelectKBest, f_classif
K=5
k_best = SelectKBest(f_classif, k=K)
k_best.fit(features, labels)
scores = k_best.scores_
print features_list
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
print K, "best features:", sorted_pairs[0:8]
selected_features = dataset.drop(columns = ['Type', 'RI', 'Na', 'Mg', 'Ca'])
print selected_features.head()
features_train, features_test, labels_train, labels_test = train_test_split(selected_features, labels, test_size = 0.2, random_state = 42)
clf.fit(features_train, labels_train)
print "Score after features selection(", K, "-Best) =", clf.score(features_test, labels_test)
print "Features RI and Na were practically useless. Features Mg and Ca aren't contributing much."
