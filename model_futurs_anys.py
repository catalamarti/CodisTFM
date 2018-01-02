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
import random

seed = 123456
Ncross = 100

data  =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors_All.txt','w');

X0 = data.values
Y0 = result.values

N    = 1000
Ncross = 100
Nfit = 1000

X = np.zeros([9000,12])
Y = np.zeros(9000)

k = 0
for ii in range(9):
	for jj in range(N):
		X[k,0:11] = X0[jj,:]
		X[k,11]   = Y0[jj,ii]
		Y[k]      = Y0[jj,ii+1]
		k = k + 1

errorSVM = 0
errorLR  = 0
errorLDA = 0
errorKNC = 0
errorDTC = 0
errorGNB = 0
errorRF  = 0

classificationDTC = np.zeros(12)
classificationRF  = np.zeros(12)

for ss in range(Ncross):
	
	print ss

	indexs = range(9000)
	indexs = random.sample(indexs,9000)

	X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X[indexs,:], Y[indexs], test_size=0.15)
	nn = len(X_v)
	 
	# Suport Vector Machine
	#clf = SVC()
	#clf.fit(X_t,Y_t)
	#Y_p = clf.predict(X_v)
	#for i in range(0,nn):
	#	errorSVM = errorSVM+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Logistic Regression
	#logreg = LogisticRegression()
	#logreg.fit(X_t, Y_t)
	#Y_p = logreg.predict(X_v)
	#for i in range(0,nn):
	#	errorLR = errorLR+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
	# Linear Discriminant Analysis
	#lda = LinearDiscriminantAnalysis()
	#lda.fit(X_t, Y_t)
	#Y_p = lda.predict(X_v)
	#for i in range(0,nn):
	#	errorLDA = errorLDA+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
	# K Neighbors Classifier
	#KNC = KNeighborsClassifier()
	#KNC.fit(X_t, Y_t)
	#Y_p = KNC.predict(X_v)
	#for i in range(0,nn):
	#	errorKNC = errorKNC+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Decision Tree Classifier
	DTC = DecisionTreeClassifier()
	DTC.fit(X_t, Y_t)
	Y_p = DTC.predict(X_v)
	for i in range(0,nn):
		errorDTC = errorDTC+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	classificationDTC = classificationDTC + DTC.feature_importances_
		
	# Gaussian Naive Bayes
	#GNB = GaussianNB()
	#GNB.fit(X_t, Y_t)
	#Y_p = GNB.predict(X_v)
	#for i in range(0,nn):
	#	errorGNB = errorGNB+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
	# Random Forest
	RFC = RandomForestClassifier()
	RFC.fit(X_t, Y_t)
	Y_p = RFC.predict(X_v)
	for i in range(0,nn):
		errorRF = errorRF+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	classificationRF = classificationRF + RFC.feature_importances_

errorSVM = errorSVM/nn/Ncross	
errorLR  = errorLR/nn/Ncross	
errorLDA = errorLDA/nn/Ncross	
errorKNC = errorKNC/nn/Ncross	
errorDTC = errorDTC/nn/Ncross	
errorGNB = errorGNB/nn/Ncross	
errorRF  = errorRF/nn/Ncross	

errr = errorSVM+errorLR+errorLDA+errorKNC+errorDTC+errorGNB+errorRF
print 'SVM error = ', errorSVM
print 'LR  error = ', errorLR
print 'LDA error = ', errorLDA
print 'KNC error = ', errorKNC
print 'DTC error = ', errorDTC
print 'GNB error = ', errorGNB
print 'RF  error = ', errorRF
print classificationDTC/Ncross
print classificationRF/Ncross
