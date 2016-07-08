from numpy import array
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from sklearn import datasets

#first point is x, second point is y
#w0*x + w1*y + w2 = 0
#expected=1

'''
ra = 100

b = [ (array([rd.uniform(0,5), rd.uniform(0,5), 1]), 1) for i in range(ra)]
t = [ (array([rd.uniform(-5,0), rd.uniform(-5,0), 1]), -1) for i in range(ra)]

#training_data = np.vstack((b,t))
'''

data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
training_data =  np.append( X, np.ones((X.shape[0], 1)), 
	axis=1 )

print(training_data)

'''
training_data = [ (array([1,2,1]), -1), (array([1,1,1]), -1), (array([2,2,1]), -1), (array([4,4,1]), 1), (array([3,4,1]), 1)
, (array([4,5,1]),1), (array([3,5,1]),1), (array([1,3,1]),-1) ]
'''


#print(training_data)
#print(len(training_data))

x=np.array([])
x = training_data.copy()
expected=[]
'''
for p in training_data:
	#x = np.vstack((x,p[0]))
	#print(x)
	try:
		x = np.vstack((x,p[0]))
	except Exception:
		x = p[0].copy()
	expected.append(p[1])

'''

for i in range(100):
	if i < 50:
		expected.append(1)
	else:
		expected.append(-1)

#print(x[:,0])
print(expected)

x1 = X[:,0]
x2 = X[:,1]

print('here = ', len(x1))

n = 100

#initialize the weights
w_new = np.array([-16.02, 20.3, 25.6])
#np.array([rd.random(), rd.random(), rd.random()])
gamma = 0.1

for i in range(100):
	for j in range(len(x1)):
		predict = w_new.dot(x[j])
		#print(predict)
		#input("Press Enter")
		w_old = w_new.copy()
		w_new = w_old + gamma*( expected[j] - 
			np.sign(predict) )*x[j]

print(w_new)

#calulate the actual output
# Generate data...
a = np.linspace(4, 8, 1000)
b = (-w_new[0]*a-w_new[2])/w_new[1]

plt.scatter(x1, x2, c=expected)
plt.plot(a, b, color='blue')
plt.show()