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
Ncross = 100

data  =   pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result = pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

f = open('./data/errors_Nvar.txt','w');

X0 = data.values
print X0.shape
Y=result.values[:,0]
print Y.shape

validation_size=0.15
N = int(len(Y)*validation_size)
errorSVM = np.zeros(Ncross*N)
e_SVM = np.zeros([2,11])
errorLR = np.zeros(Ncross*N)
e_LR = np.zeros([2,11])
errorLDA = np.zeros(Ncross*N)
e_LDA = np.zeros([2,11])
errorKNC = np.zeros(Ncross*N)
e_KNC = np.zeros([2,11])
errorDTC = np.zeros(Ncross*N)
e_DTC = np.zeros([2,11])
errorGNB = np.zeros(Ncross*N)
e_GNB = np.zeros([2,11])
errorRF = np.zeros(Ncross*N)
e_RF = np.zeros([2,11])
classificationDTC = [7, 1, 9, 5, 6, 2, 3, 4, 0, 10, 8]
classificationRFC = [7, 9, 1, 6, 2, 5, 3, 4, 0, 10, 8]

for ii in range(2):
	
	if (ii==0):
		cc = classificationDTC
	else:
		cc = classificationRFC

	for c in range(11):
		
		print cc[0:c+1]
		X = X0[:,cc[0:c+1]]
		print X.shape
	
		k=0
		
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
		
			k=k+N
		
		e_SVM[ii,c] = sum(errorSVM)/Ncross/N
		e_LR[ii,c]  = sum(errorLR)/Ncross/N
		e_LDA[ii,c] = sum(errorLDA)/Ncross/N
		e_KNC[ii,c] = sum(errorKNC)/Ncross/N
		e_DTC[ii,c] = sum(errorDTC)/Ncross/N
		e_GNB[ii,c] = sum(errorGNB)/Ncross/N
		e_RF[ii,c]  = sum(errorRF)/Ncross/N

	for err in e_SVM[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_LR[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_LDA[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_KNC[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_DTC[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_GNB[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
	for err in e_RF[ii,:]:
		f.write('%.4f ' % err)
	f.write('\n')
f.close()

