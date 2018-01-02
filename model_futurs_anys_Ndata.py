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

data  =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors_AllvsNdata2.txt','w');

X0 = data.values
Y0 = result.values

N    = 1000
Ncross = 10

X = np.zeros([9000,12])
Y = np.zeros(9000)

k = 0
for ii in range(9):
	for jj in range(N):
		X[k,0:11] = X0[jj,:]
		X[k,11]   = Y0[jj,ii]
		Y[k]      = Y0[jj,ii+1]
		k = k + 1

nnfit = 10
Nfit_v = np.logspace(2.3,3.90309,num=nnfit)
for ii in range(len(Nfit_v)):
	Nfit_v[ii] = int(round(Nfit_v[ii]))
Nfit_v = np.int_(Nfit_v)
print Nfit_v

errorSVM = np.zeros(nnfit)
errorLR  = np.zeros(nnfit)
errorLDA = np.zeros(nnfit)
errorKNC = np.zeros(nnfit)
errorDTC = np.zeros(nnfit)
errorGNB = np.zeros(nnfit)
errorRF  = np.zeros(nnfit)

for k in range(nnfit):

	Nfit = Nfit_v[k]
	X_t0, X_v, Y_t0, Y_v = model_selection.train_test_split(X, Y, test_size=0.1)

	for ss in range(Ncross):
		
		print k, ss
	
		indexs = range(8000)
		indexs = random.sample(indexs,Nfit)
	
		X_t = X_t0[indexs,:]
		Y_t = Y_t0[indexs]
		nn = len(X_v)
		 
		# Suport Vector Machine
		clf = SVC()
		clf.fit(X_t,Y_t)
		Y_p = clf.predict(X_v)
		for i in range(0,nn):
			errorSVM[k] = errorSVM[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
		# Logistic Regression
		logreg = LogisticRegression()
		logreg.fit(X_t, Y_t)
		Y_p = logreg.predict(X_v)
		for i in range(0,nn):
			errorLR[k] = errorLR[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
			
		# Linear Discriminant Analysis
		lda = LinearDiscriminantAnalysis()
		lda.fit(X_t, Y_t)
		Y_p = lda.predict(X_v)
		for i in range(0,nn):
			errorLDA[k] = errorLDA[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
			
		# K Neighbors Classifier
		KNC = KNeighborsClassifier()
		KNC.fit(X_t, Y_t)
		Y_p = KNC.predict(X_v)
		for i in range(0,nn):
			errorKNC[k] = errorKNC[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
		# Decision Tree Classifier
		DTC = DecisionTreeClassifier()
		DTC.fit(X_t, Y_t)
		Y_p = DTC.predict(X_v)
		for i in range(0,nn):
			errorDTC[k] = errorDTC[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
			
		# Gaussian Naive Bayes
		GNB = GaussianNB()
		GNB.fit(X_t, Y_t)
		Y_p = GNB.predict(X_v)
		for i in range(0,nn):
			errorGNB[k] = errorGNB[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
		# Random Forest
		RFC = RandomForestClassifier()
		RFC.fit(X_t, Y_t)
		Y_p = RFC.predict(X_v)
		for i in range(0,nn):
			errorRF[k] = errorRF[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
errorSVM = errorSVM/nn/Ncross	
errorLR  = errorLR/nn/Ncross	
errorLDA = errorLDA/nn/Ncross	
errorKNC = errorKNC/nn/Ncross	
errorDTC = errorDTC/nn/Ncross	
errorGNB = errorGNB/nn/Ncross	
errorRF  = errorRF/nn/Ncross	

errr = errorSVM+errorLR+errorLDA+errorKNC+errorDTC+errorGNB+errorRF
errr = errr/7
print 'SVM error = ', errorSVM
print 'LR  error = ', errorLR
print 'LDA error = ', errorLDA
print 'KNC error = ', errorKNC
print 'DTC error = ', errorDTC
print 'GNB error = ', errorGNB
print 'RF  error = ', errorRF

for err in Nfit_v:
	f.write('%.4f ' % err)
f.write('\n')
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
for err in errr:
	f.write('%.4f ' % err)
f.write('\n')
f.close()
