import pandas
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

NS = np.logspace(1.5,2.9,10)
NS=np.int_(NS)

ET=np.zeros(len(NS))
	
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(len(NS)):
	clf = LinearDiscriminantAnalysis(solver='svd')
	clf.fit(X_t[0:NS[i]-1][0:-1],Y_t[0:NS[i]-1][0:-1])
	Y_p = clf.predict(X_v)
	error= np.zeros(N)
	for ii in range(0,N):
		error[ii] = (float(Y_v[ii]-Y_p[ii])/Y_v[ii])**2
	ET[i]=np.sqrt(sum(error))/N
plt.semilogx(NS,ET)
plt.show()