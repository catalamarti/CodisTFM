import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

seed = 123456
Ncross = 1000

data  =   pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors.txt','w');

X = data.values
print X.shape
Y=result.values[:,0]
print Y.shape

validation_size=0.15
N = int(len(Y)*validation_size)
errorSVM = np.zeros(Ncross*N)
errorLR = np.zeros(Ncross*N)
errorLDA = np.zeros(Ncross*N)
errorKNC = np.zeros(Ncross*N)
errorDTC = np.zeros(Ncross*N)
errorGNB = np.zeros(Ncross*N)
errorRF = np.zeros(Ncross*N)
classificationDTC = np.zeros(11)
classificationRFC = np.zeros(11)

k=0
print N

for nn in range(Ncross):

	X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X, Y, test_size=validation_size)
	
	# Suport Vector Machine
	clf = SVC()
	clf.fit(X_t,Y_t)
	Y_p = clf.predict(X_v)
	for i in range(0,N):
		errorSVM[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Logistic Regression
	logreg = LogisticRegression()
	logreg.fit(X_t, Y_t)
	Y_p = logreg.predict(X_v)
	for i in range(0,N):
		errorLR[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Linear Discriminant Analysis
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_t, Y_t)
	Y_p = lda.predict(X_v)
	for i in range(0,N):
		errorLDA[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# K Neighbors Classifier
	KNC = KNeighborsClassifier()
	KNC.fit(X_t, Y_t)
	Y_p = KNC.predict(X_v)
	for i in range(0,N):
		errorKNC[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]

	# Decision Tree Classifier
	DTC = DecisionTreeClassifier()
	DTC.fit(X_t, Y_t)
	Y_p = DTC.predict(X_v)
	for i in range(0,N):
		errorDTC[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	classificationDTC = classificationDTC + DTC.feature_importances_
	
	# Gaussian Naive Bayes
	GNB = GaussianNB()
	GNB.fit(X_t, Y_t)
	Y_p = GNB.predict(X_v)
	for i in range(0,N):
		errorGNB[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Random Forest
	RFC = RandomForestClassifier()
	RFC.fit(X_t, Y_t)
	Y_p = RFC.predict(X_v)
	for i in range(0,N):
		errorRF[k+i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	classificationRFC = classificationRFC + RFC.feature_importances_

	k=k+N

for err in errorSVM:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorLR:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorLDA:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorKNC:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorDTC:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorGNB:
	f.write('%.4f ' % err)
f.write('\n')
for err in errorRF:
	f.write('%.4f ' % err)
f.write('\n')
f.close()

print sum(errorSVM)/N/Ncross
print sum(errorLR)/N/Ncross
print sum(errorLDA)/N/Ncross
print sum(errorKNC)/N/Ncross
print sum(errorDTC)/N/Ncross
print sum(errorGNB)/N/Ncross
print sum(errorRF)/N/Ncross

print classificationDTC/Ncross
print classificationRFC/Ncross
