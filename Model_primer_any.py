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

X = X0[:,[7, 1, 9, 5, 6, 2, 3, 4]]
X, X_v2, Y, Y_v2 = model_selection.train_test_split(X, Y, test_size=0.1)

validation_size=0.11111
N = 100

errorLDA = np.zeros(N)
	
mean_error = 0
max_error = 0
std_error = 0
		
for nn in range(Ncross):
	
	X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X, Y, test_size=validation_size)
			
	# Linear Discriminant Analysis
	lda = LinearDiscriminantAnalysis()
	lda.fit(X_t, Y_t)
	Y_p = lda.predict(X_v)
	for i in range(0,N):
		errorLDA[i] = abs(float(Y_v[i]-Y_p[i]))/Y_v[i]
	mean_error = sum(errorLDA)/N + mean_error
	max_error = max(errorLDA) + max_error
	std_error = np.std(errorLDA) + std_error
		
print mean_error/Ncross
print max_error/Ncross
print std_error/Ncross

seed = 123456

lda = LinearDiscriminantAnalysis()
lda.fit(X, Y)
Y_p2 = lda.predict(X_v2)
for i in range(0,N):
	errorLDA[i] = abs(float(Y_v2[i]-Y_p2[i]))/Y_v2[i]

print sum(errorLDA)/N
print max(errorLDA)
print np.std(errorLDA)
