#import libraries
import numpy as np
from math import e
import matplotlib
import matplotlib.pyplot as plt

### ---------------- DATA COLLECTION (case specific) -----------------------------------

#set directory
import os
path = 'C:\\Users\\smook\\Downloads\\ex2-octave'
os.chdir(path)

#gather data
data = np.genfromtxt(fname='ex2data1.txt',delimiter=',')
xt = data[:,0:2]
yt = data[:,2]
m = yt.size
xb = np.column_stack((np.ones(shape=(m)), xt))
yb = np.reshape(yt,(m,1))

### -------------------------------------------------------------------------------------

### ---------------- SET VARIABLES (required) -------------------------------------------

x = xb
y = yb
learning_rate = 0.001 #(alpha)
iterations = 500000
initial_theta = np.zeros(shape=(3,1))

print(x)

### -------------------------------------------------------------------------------------

### ---------------- ALGORITHM ----------------------------------------------------------

m = y.size

#plot data
fig, ax = plt.subplots()
ax.scatter(xt[:,0], xt[:,1], c=y, cmap = 'copper')
plt.show(block=False)

np.seterr(divide = 'ignore',over = 'ignore',invalid = 'ignore')

#cost function
def costFunction(theta):
    hypothesis = 1/(1+e**(-np.dot(x,theta)))
    J = 1/m*(-y*np.log(hypothesis)-(1-y)*np.log(1-hypothesis)).sum()
    return J

print('The cost with theta set to 0 is: '+str(costFunction(np.zeros(shape=(3,1)))))

#gradient descent
def gradientDescent(theta,alpha,num_iters):
    J_history = np.zeros(shape=(num_iters))
    for i in range(1,num_iters+1):
        hypothesis = 1/(1+e**(-np.dot(x,theta)))
        theta = theta - (1/m)*alpha*( x.T.dot(hypothesis - y))
        J_history[i-1] = costFunction(theta)
    return theta, J_history

theta, J_history = gradientDescent(initial_theta,learning_rate,iterations)

print(theta)

#plot decision boundary
x2 = np.linspace(25,100,100)
y2 = -theta[0,0]/theta[2,0] + (-theta[1,0])/theta[2,0]*x2
ax.plot(x2,y2)

plt.show(block = True)

print('Final Hypothesis: sigmoid('+str(theta[1,0])+'*x1 + '+str(theta[2,0])+'*x2 + '+str(theta[0,0]))

### --------------------------------------------------------------------------------------

