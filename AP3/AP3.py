import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score


names=['id number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', sep=",", names=names, na_values=['?'])
data = data.dropna()

feature_cols = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
X = data[feature_cols]
y = (data["Class"]
         .replace("2",0)
         .replace("4",1)
         .values.tolist())

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("All Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("All Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("All Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Clump Thickness']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Clump Thickness Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Clump Thickness Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Clump Thickness Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Uniformity of Cell Size']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Uniformity of Cell Size Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Uniformity of Cell Size Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Uniformity of Cell Size Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Uniformity of Cell Shape']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Uniformity of Cell Shape Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Uniformity of Cell Shape Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Uniformity of Cell Shape Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Marginal Adhesion']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Marginal Adhesion Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Marginal Adhesion Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Marginal Adhesion Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Single Epithelial Cell Size']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Single Epithelial Cell Size Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Single Epithelial Cell Size Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Single Epithelial Cell Size Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Bare Nuclei']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Bare Nuclei Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Bare Nuclei Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Bare Nuclei Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Bland Chromatin']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Bland Chromatin Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Bland Chromatin Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Bland Chromatin Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Normal Nucleoli']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Normal Nucleoli Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Normal Nucleoli Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Normal Nucleoli Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))

feature_cols = ['Mitoses']
X = data[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

print (clf.coef_)

y_pred = clf.predict(X_test)

print ("Mitoses Precision: %1.3f" %(precision_score(y_test, y_pred) * 100.00))
print ("Mitoses Accuracy:  %1.3f " %(accuracy_score(y_test, y_pred) * 100.00))
print ("Mitoses Recall:   %1.3f" %(recall_score(y_test, y_pred) * 100.00))
