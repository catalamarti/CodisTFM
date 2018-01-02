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
Ncross = 10

data  =   pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors_AllvsNdata1.txt','w');

X0 = data.values
print X0.shape
Y0 =result.values[:,0]
print Y0.shape

nnfit = 10
Nfit_v = np.logspace(1.7,2.954,num=nnfit)
for ii in range(len(Nfit_v)):
	Nfit_v[ii] = int(round(Nfit_v[ii]))
Nfit_v = np.int_(Nfit_v)

validation_size=0.1
errorSVM = np.zeros(nnfit)
errorLR = np.zeros(nnfit)
errorLDA = np.zeros(nnfit)
errorKNC = np.zeros(nnfit)
errorDTC = np.zeros(nnfit)
errorGNB = np.zeros(nnfit)
errorRF = np.zeros(nnfit)

for nn in range(Ncross):

	X_t0, X_v, Y_t0, Y_v = model_selection.train_test_split(X0, Y0, test_size=validation_size)
	N = len(Y_v)
	print nn, N

	for k in range(nnfit):

		Nfit = Nfit_v[k]
		indexs = range(len(X_t0))
		indexs = random.sample(indexs,Nfit)
		X_t = X_t0[indexs,:]	
		Y_t = Y_t0[indexs]			

		# Suport Vector Machine
		clf = SVC()
		clf.fit(X_t,Y_t)
		Y_p = clf.predict(X_v)
		for i in range(0,N):
			errorSVM[k] = errorSVM[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
		# Logistic Regression
		logreg = LogisticRegression()
		logreg.fit(X_t, Y_t)
		Y_p = logreg.predict(X_v)
		for i in range(0,N):
			errorLR[k] = errorLR[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
		# Linear Discriminant Analysis
		lda = LinearDiscriminantAnalysis()
		lda.fit(X_t, Y_t)
		Y_p = lda.predict(X_v)
		for i in range(0,N):
			errorLDA[k] = errorLDA[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
		# K Neighbors Classifier
		KNC = KNeighborsClassifier()
		KNC.fit(X_t, Y_t)
		Y_p = KNC.predict(X_v)
		for i in range(0,N):
			errorKNC[k] = errorKNC[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
		# Decision Tree Classifier
		DTC = DecisionTreeClassifier()
		DTC.fit(X_t, Y_t)
		Y_p = DTC.predict(X_v)
		for i in range(0,N):
			errorDTC[k] = errorDTC[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]

		# Gaussian Naive Bayes
		GNB = GaussianNB()
		GNB.fit(X_t, Y_t)
		Y_p = GNB.predict(X_v)
		for i in range(0,N):
			errorGNB[k] = errorGNB[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	
		# Random Forest
		RFC = RandomForestClassifier()
		RFC.fit(X_t, Y_t)
		Y_p = RFC.predict(X_v)
		for i in range(0,N):
			errorRF[k] = errorRF[k]+abs(float(Y_v[i]-Y_p[i]))/Y_v[i]

errorSVM = errorSVM/N/Ncross
errorLR  = errorLR/N/Ncross
errorLDA = errorLDA/N/Ncross
errorKNC = errorKNC/N/Ncross
errorDTC = errorDTC/N/Ncross
errorGNB = errorGNB/N/Ncross
errorRF  = errorRF/N/Ncross

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
f.close()

print errorSVM
print errorLR
print errorLDA
print errorKNC
print errorDTC
print errorGNB
print errorRF
