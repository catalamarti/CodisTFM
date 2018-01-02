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

Nin  = 8
Nout = 9 - Nin
N    = 1000
Ncross = 1
Nfit = N*Nin

X = np.zeros([N*Nin,12])
Y = np.zeros(N*Nin)

k = 0
for ii in range(Nin):
	for jj in range(N):
		X[k,0:11] = X0[jj,:]
		X[k,11]   = Y0[jj,ii]
		Y[k]      = Y0[jj,ii+1] 
		k = k + 1

errorSVM = np.zeros(Nout)
errorLR  = np.zeros(Nout)
errorLDA = np.zeros(Nout)
errorKNC = np.zeros(Nout)
errorDTC = np.zeros(Nout)
errorGNB = np.zeros(Nout)
errorRF  = np.zeros(Nout)

classificationDTC = np.zeros(12)
classificationRF  = np.zeros(12)

for ss in range(Ncross):
	
	indexs = range(N*Nin)
	indexs = random.sample(indexs,Nfit)
	
	X_t = X[indexs,:]
	Y_t = Y[indexs]
	X_v = np.zeros([Nout*N,12])
	Y_v = np.zeros(Nout*N)
	k = 0
	for ii in range(Nout):
		for jj in range(N):
			X_v[k,0:11] = X0[jj,:]
			X_v[k,11]   = Y0[jj,ii+Nin]
			Y_v[k]      = Y0[jj,ii+Nin+1]
			k = k + 1
	 
	# Suport Vector Machine
	clf = SVC()
	clf.fit(X_t,Y_t)
	Y_p = clf.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorSVM[j] = errorSVM[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
	
	# Logistic Regression
	logreg = LogisticRegression()
	logreg.fit(X_t, Y_t)
	Y_p = logreg.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorLR[j] = errorLR[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
		
	# Linear Discriminant Analysis
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_t, Y_t)
	Y_p = lda.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorLDA[j] = errorLDA[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
		
	# K Neighbors Classifier
	KNC = KNeighborsClassifier()
	KNC.fit(X_t, Y_t)
	Y_p = KNC.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorKNC[j] = errorKNC[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
	
	# Decision Tree Classifier
	DTC = DecisionTreeClassifier()
	DTC.fit(X_t, Y_t)
	Y_p = DTC.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorDTC[j] = errorDTC[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
	classificationDTC = classificationDTC + DTC.feature_importances_
		
	# Gaussian Naive Bayes
	GNB = GaussianNB()
	GNB.fit(X_t, Y_t)
	Y_p = GNB.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorGNB[j] = errorGNB[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
	
	# Random Forest
	RFC = RandomForestClassifier()
	RFC.fit(X_t, Y_t)
	Y_p = RFC.predict(X_v)
	k = 0
	for j in range(Nout):
		for i in range(0,N):
			errorRF[j] = errorRF[j]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
			k = k + 1
	classificationRF = classificationRF + RFC.feature_importances_

errorSVM = errorSVM/N/Ncross	
errorLR  = errorLR/N/Ncross	
errorLDA = errorLDA/N/Ncross	
errorKNC = errorKNC/N/Ncross	
errorDTC = errorDTC/N/Ncross	
errorGNB = errorGNB/N/Ncross	
errorRF  = errorRF/N/Ncross
errr = errorSVM+errorLR+errorLDA+errorKNC+errorDTC+errorGNB+errorRF

classificationDTC = classificationDTC/Ncross
classificationRF  = classificationRF/Ncross

print 'SVM error = ', errorSVM
print 'LR  error = ', errorLR
print 'LDA error = ', errorLDA
print 'KNC error = ', errorKNC
print 'DTC error = ', errorDTC
print 'GNB error = ', errorGNB
print 'RF  error = ', errorRF
print 'Meanerror = ', errr/7
print 'SVM error = ', sum(errorSVM)/Nout
print 'LR  error = ', sum(errorLR)/Nout
print 'LDA error = ', sum(errorLDA)/Nout
print 'KNC error = ', sum(errorKNC)/Nout
print 'DTC error = ', sum(errorDTC)/Nout
print 'GNB error = ', sum(errorGNB)/Nout
print 'RF  error = ', sum(errorRF)/Nout
print classificationDTC
print classificationRF
