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

f = open('./data/errors_Nvar_model21.txt','w');

X0 = data.values
Y0 = result.values

Nin  = 8
Nout = 1
N    = 1000
Ncross = 1
Nfit = 8000

X = np.zeros([N*Nin,12])
Y = np.zeros(N*Nin)

k = 0
for ii in range(Nin):
	for jj in range(N):
		X[k,0:11] = X0[jj,:]
		X[k,11]   = Y0[jj,ii]
		Y[k]      = Y0[jj,ii+1] 
		k = k + 1

X_v0 = np.zeros([Nout*N,12])
Y_v0 = np.zeros(Nout*N)
for jj in range(N):
	X_v0[jj,0:11] = X0[jj,:]
	X_v0[jj,11]   = Y0[jj,8]
	Y_v0[jj]      = Y0[jj,9]

ordreDTC = [11, 5, 10, 0, 1, 4, 7, 3, 6, 2, 9, 8]
ordreRFC = [11, 6, 3, 2, 9, 4, 1, 5, 7, 10, 0, 8]

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
			
			X_t = X[:,classificacio[0:var+1]]
			Y_t = Y
			X_v = X_v0[:,classificacio[0:var+1]]
			Y_v = Y_v0			

			# Suport Vector Machine
			clf = SVC()
			clf.fit(X_t,Y_t)
			Y_p = clf.predict(X_v)
			for i in range(0,N):
				errorSVM[var,order] = errorSVM[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
			
			# Logistic Regression
			logreg = LogisticRegression()
			logreg.fit(X_t, Y_t)
			Y_p = logreg.predict(X_v)
			for i in range(0,N):
				errorLR[var,order] = errorLR[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
				
			# Linear Discriminant Analysis
			lda = LinearDiscriminantAnalysis()
			lda.fit(X_t, Y_t)
			Y_p = lda.predict(X_v)
			for i in range(0,N):
				errorLDA[var,order] = errorLDA[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
			# K Neighbors Classifier
			KNC = KNeighborsClassifier()
			KNC.fit(X_t, Y_t)
			Y_p = KNC.predict(X_v)
			for i in range(0,N):
				errorKNC[var,order] = errorKNC[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
			# Decision Tree Classifier
			DTC = DecisionTreeClassifier()
			DTC.fit(X_t, Y_t)
			Y_p = DTC.predict(X_v)
			for i in range(0,N):
				errorDTC[var,order] = errorDTC[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
		
			# Gaussian Naive Bayes
			GNB = GaussianNB()
			GNB.fit(X_t, Y_t)
			Y_p = GNB.predict(X_v)
			for i in range(0,N):
				errorGNB[var,order] = errorGNB[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
			# Random Forest
			RFC = RandomForestClassifier()
			RFC.fit(X_t, Y_t)
			Y_p = RFC.predict(X_v)
			for i in range(0,N):
				errorRF[var,order] = errorRF[var,order]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]

errorSVM = errorSVM/N/Ncross	
errorLR  = errorLR/N/Ncross	
errorLDA = errorLDA/N/Ncross	
errorKNC = errorKNC/N/Ncross	
errorDTC = errorDTC/N/Ncross	
errorGNB = errorGNB/N/Ncross	
errorRF  = errorRF/N/Ncross

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
