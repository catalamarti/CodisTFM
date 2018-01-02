import pandas
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import numpy as np

data =  pandas.read_excel(open('./data/results.xlsx','rb'), sheet_name=0);
result =  pandas.read_excel(open('./data/results.xlsx','rb'), sheet_name=1);

X = data.values
Y=result.values[:,0]

validation_size=0.2
seed = 7
X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

N = len(Y_v)

CC = np.linspace(5,10,6)
#t = np.logspace(-7,-3,10)
CC=np.int_(CC)

cc=np.zeros(len(CC))
et=np.zeros(len(CC))
index=0

ET=np.zeros(len(CC))
	
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
for i in range(len(CC)):
	clf = GaussianNB(priors=CC[i])
	clf.fit(X_t,Y_t)
	Y_p = clf.predict(X_v)
	error= np.zeros(N)
	for ii in range(0,N):
		error[ii] = (float(Y_v[ii]-Y_p[ii])/Y_v[ii])**2
	ET[i]=np.sqrt(sum(error))/N
	et[index]=np.sqrt(sum(error))/N
	cc[index]=CC[i]
	index=index+1
#plt.imshow(ET, interpolation='nearest', cmap=plt.cm.hot)
#plt.colorbar()
plt.semilogx(CC,ET)
plt.show()

print ET
id = np.argmin(ET)
print et[id]
print cc[id]

clf = GaussianNB(priors=CC[id])
clf.fit(X_t,Y_t)
Y_p = clf.predict(X_v)
error= np.zeros(N)
for ii in range(0,N):
	error[ii] = (float(Y_v[ii]-Y_p[ii])/Y_v[ii])**2

print np.sqrt(sum(error))/N
print max(error)
print np.std(error)