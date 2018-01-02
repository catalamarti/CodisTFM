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
Ncross = 1

data  =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors_Nvar_model22.txt','w');

X0 = data.values
Y0 = result.values

N    = 1000
Ncross = 5
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

ordreDTC = [11, 6, 7, 3, 1, 2, 10, 4, 0, 5, 9, 8]
ordreRFC = [11, 6, 3, 7, 1, 4, 5, 2, 10, 0, 9, 8]

errorSVM = np.zeros([12,2])
errorLR  = np.zeros([12,2])
errorLDA = np.zeros([12,2])
errorKNC = np.zeros([12,2])
errorDTC = np.zeros([12,2])
errorGNB = np.zeros([12,2])
errorRF  = np.zeros([12,2])

for ss in range(Ncross):

	for var in range(12):
		for order in range(2):
	
			print ss, var, order
			if order == 0:
				classificacio = ordreDTC
			else:
				classificacio = ordreRFC

			indexs = range(9000)
			indexs = random.sample(indexs,9000)
			X = X[indexs,:]
			Y = Y[indexs]
			X_t = X[0:8000,classificacio[0:var+1]]
			Y_t = Y[0:8000]
			X_v = X[8000:9000,classificacio[0:var+1]]
			Y_v = Y[8000:9000]
			nn = len(X_v)
	 
			# Suport Vector Machine
			clf = SVC()
			clf.fit(X_t,Y_t)
			Y_p = clf.predict(X_v)
			for i in range(0,nn):
				errorSVM[var,order] = errorSVM[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
			# Logistic Regression
			logreg = LogisticRegression()
			logreg.fit(X_t, Y_t)
			Y_p = logreg.predict(X_v)
			for i in range(0,nn):
				errorLR[var,order] = errorLR[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
			# Linear Discriminant Analysis
			lda = LinearDiscriminantAnalysis()
			lda.fit(X_t, Y_t)
			Y_p = lda.predict(X_v)
			for i in range(0,nn):
				errorLDA[var,order] = errorLDA[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
			# K Neighbors Classifier
			KNC = KNeighborsClassifier()
			KNC.fit(X_t, Y_t)
			Y_p = KNC.predict(X_v)
			for i in range(0,nn):
				errorKNC[var,order] = errorKNC[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
			# Decision Tree Classifier
			DTC = DecisionTreeClassifier()
			DTC.fit(X_t, Y_t)
			Y_p = DTC.predict(X_v)
			for i in range(0,nn):
				errorDTC[var,order] = errorDTC[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
			# Gaussian Naive Bayes
			GNB = GaussianNB()
			GNB.fit(X_t, Y_t)
			Y_p = GNB.predict(X_v)
			for i in range(0,nn):
				errorGNB[var,order] = errorGNB[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
			# Random Forest
			RFC = RandomForestClassifier()
			RFC.fit(X_t, Y_t)
			Y_p = RFC.predict(X_v)
			for i in range(0,nn):
				errorRF[var,order] = errorRF[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]

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

for ii in range(2):
	for err in errorSVM[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorLR[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorLDA[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorKNC[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorDTC[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorGNB[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in errorRF[:,ii]:
		f.write('%.4f ' % err)
	f.write('\n')
f.close()
