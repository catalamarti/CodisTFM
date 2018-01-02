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

f = open('./data/errors_AllvsNdata.txt','w');

X0 = data.values
Y0 = result.values

Nin  = 8
Nout = 9 - Nin
N    = 1000
Ncross = 10

X = np.zeros([N*Nin,12])
Y = np.zeros(N*Nin)

k = 0
for ii in range(Nin):
	for jj in range(N):
		X[k,0:11] = X0[jj,:]
		X[k,11]   = Y0[jj,ii]
		Y[k]      = Y0[jj,ii+1] 
		k = k + 1

nnfit = 10
Nfit_v = np.logspace(2.0,3.90309,num=nnfit)
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

for kk in range(nnfit):
	
	Nfit = Nfit_v[kk]
	print Nfit

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
				errorSVM[kk] = errorSVM[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
		
		# Logistic Regression
		logreg = LogisticRegression()
		logreg.fit(X_t, Y_t)
		Y_p = logreg.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorLR[kk] = errorLR[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
			
		# Linear Discriminant Analysis
		lda = LinearDiscriminantAnalysis()
		lda.fit(X_t, Y_t)
		Y_p = lda.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorLDA[kk] = errorLDA[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
			
		# K Neighbors Classifier
		KNC = KNeighborsClassifier()
		KNC.fit(X_t, Y_t)
		Y_p = KNC.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorKNC[kk] = errorKNC[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
		
		# Decision Tree Classifier
		DTC = DecisionTreeClassifier()
		DTC.fit(X_t, Y_t)
		Y_p = DTC.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorDTC[kk] = errorDTC[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
			
		# Gaussian Naive Bayes
		GNB = GaussianNB()
		GNB.fit(X_t, Y_t)
		Y_p = GNB.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorGNB[kk] = errorGNB[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
		
		# Random Forest
		RFC = RandomForestClassifier()
		RFC.fit(X_t, Y_t)
		Y_p = RFC.predict(X_v)
		k = 0
		for j in range(Nout):
			for i in range(0,N):
				errorRF[kk] = errorRF[kk]+abs(float(Y_v[k]-Y_p[k]))/Y_v[k]
				k = k + 1
	
errorSVM = errorSVM/N/Ncross	
errorLR  = errorLR/N/Ncross	
errorLDA = errorLDA/N/Ncross	
errorKNC = errorKNC/N/Ncross	
errorDTC = errorDTC/N/Ncross	
errorGNB = errorGNB/N/Ncross	
errorRF  = errorRF/N/Ncross	
errr = errorSVM+errorLR+errorLDA+errorKNC+errorDTC+errorGNB+errorRF
errr = errr/7
print 'SVM error = ', errorSVM
print 'LR  error = ', errorLR
print 'LDA error = ', errorLDA
print 'KNC error = ', errorKNC
print 'DTC error = ', errorDTC
print 'GNB error = ', errorGNB
print 'RF  error = ', errorRF
print 'Meanerror = ', errr/7

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
