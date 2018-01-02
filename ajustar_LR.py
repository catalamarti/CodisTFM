import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import numpy as np

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k/ncol, k%ncol

seed = 123456

data =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=0);
result =  pandas.read_excel(open('./data/results_new.xlsx','rb'), sheet_name=1);

X = data.values
Y=result.values[:,0]

validation_size=0.1
X, X_v2, Y, Y_v2 = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
validation_size=0.1111
X_t, X_v, Y_t, Y_v = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

N = len(Y_v)

CC = np.logspace(-1,6,10)
et=np.zeros((len(CC),5))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(len(CC)):
	clf_1 = LogisticRegression(C=CC[i],solver='lbfgs')
	clf_2 = LogisticRegression(C=CC[i],solver='newton-cg')
	clf_3 = LogisticRegression(C=CC[i],solver='liblinear')
	clf_4 = LogisticRegression(C=CC[i],solver='sag')
	clf_5 = LogisticRegression(C=CC[i],solver='saga')
	clf_1.fit(X_t,Y_t)
	clf_2.fit(X_t,Y_t)
	clf_3.fit(X_t,Y_t)
	clf_4.fit(X_t,Y_t)
	clf_5.fit(X_t,Y_t)
	Y_p_1 = clf_1.predict(X_v)
	Y_p_2 = clf_2.predict(X_v)
	Y_p_3 = clf_3.predict(X_v)
	Y_p_4 = clf_4.predict(X_v)
	Y_p_5 = clf_5.predict(X_v)
	error_1= np.zeros(N)
	error_2= np.zeros(N)
	error_3= np.zeros(N)
	error_4= np.zeros(N)
	error_5= np.zeros(N)
	for ii in range(0,N):
		error_1[ii] = abs(float(Y_v[ii]-Y_p_1[ii]))/Y_v[ii]
		error_2[ii] = abs(float(Y_v[ii]-Y_p_2[ii]))/Y_v[ii]
		error_3[ii] = abs(float(Y_v[ii]-Y_p_3[ii]))/Y_v[ii]
		error_4[ii] = abs(float(Y_v[ii]-Y_p_4[ii]))/Y_v[ii]
		error_5[ii] = abs(float(Y_v[ii]-Y_p_5[ii]))/Y_v[ii]
	et[i,0]=sum(error_1)/N
	et[i,1]=sum(error_2)/N
	et[i,2]=sum(error_3)/N
	et[i,3]=sum(error_4)/N
	et[i,4]=sum(error_5)/N
	print et[i,:]
	
plt.semilogx(CC,et[:,0])
plt.semilogx(CC,et[:,1])
plt.semilogx(CC,et[:,2])
plt.semilogx(CC,et[:,3])
plt.semilogx(CC,et[:,4])
plt.show()

id =find_min_idx(et)
print id
if id[1]==0:
	model='lbfgs'
else:
	if id[1]==1:
		model='newton-cg'
	else:
		if id[1]==2:
			model='liblinear'
		else:
			if id[1]==3:
				model='sag'
			else:
				model='saga'
clf = LogisticRegression(C=CC[id[0]],solver=model)
clf.fit(X_t,Y_t)
Y_p = clf.predict(X_v)
Y_p2 = clf.predict(X_v2)
error= np.zeros(N)
error2= np.zeros(N)
for ii in range(0,N):
	error[ii] = abs(float(Y_v[ii]-Y_p[ii]))/Y_v[ii]
	error2[ii] = abs(float(Y_v2[ii]-Y_p2[ii]))/Y_v2[ii]
ET=sum(error)/N
print ET
print max(error)
print np.std(error)
ET2=sum(error2)/N
print ET2
print max(error2)
print np.std(error2)