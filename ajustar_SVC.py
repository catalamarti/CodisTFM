import pandas
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import model_selection
import numpy as np

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k/ncol, k%ncol

def SVC_model(model,Ncross,X,Y,validation_size):
	X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X, Y, test_size=validation_size)
	N=len(X_v)
	print N, Ncross
	error=np.zeros(N*Ncross)
	print len(error)
	k=0
	for m in range(Ncross):
		clf = model
		clf.fit(X_t,Y_t)
		Y_p = clf.predict(X_v)
		for ii in range(0,N):
			error[ii+k] = abs(float(Y_v[ii]-Y_p[ii]))/Y_v[ii]
		k=k+N		
	ET=sum(error)/N
	return ET,max(error),np.std(error)

seed = 123456

data =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

X = data.values
Y=result.values[:,0]

validation_size=0.1
X, X_v2, Y, Y_v2 = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

validation_size=0.1111
Ncross=10

CC = np.logspace(0,5,11)
g = np.logspace(-8,1,10)
	
fig = plt.figure()
for d in range (1,2):
	ET=np.zeros((len(CC),len(g)))
	ME=np.zeros((len(CC),len(g)))
	STD=np.zeros((len(CC),len(g)))
	ax = fig.add_subplot(1,1,d)
	ax.set_aspect('equal')
	for i in range(len(CC)):
		for j in range(len(g)):
			ET[i][j],ME[i][j],STD[i][j]=SVC_model(SVC(C=CC[i],gamma=g[j], degree=d, kernel='linear'), Ncross, X, Y, validation_size)
	plt.imshow(ET, interpolation='nearest', cmap=plt.cm.hot)
	plt.colorbar()
plt.show()
i,j = find_min_idx(ET)
print CC[i], g[j]
print ET[i][j], ME[i][j], STD[i][j]

N=len(X_v2)
clf = SVC(C=CC[i],gamma=g[j])
clf.fit(X,Y)
Y_p2 = clf.predict(X_v2)
error= np.zeros(N)
for ii in range(0,N):
	error[ii] = abs(float(Y_v2[ii]-Y_p2[ii]))/Y_v2[ii]
ET=sum(error)/N
print ET
print max(error)
print np.std(error)
