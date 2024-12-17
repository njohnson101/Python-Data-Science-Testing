#import libraries
from time import sleep
wait = sleep 
import numpy as np
import matplotlib.pyplot as plt

### ---------------- DATA COLLECTION (case specific) -----------------------------------

#set directory
import os
path = 'C:\\Users\\smook\\Downloads\\ex1-octave'
os.chdir(path)

#gather data
data = np.genfromtxt(fname='ex1data1.txt',delimiter=',')
xt = data[:,0]
yt = data[:,1]
m = yt.size
xb = np.column_stack((np.ones(shape=(m)), xt))
yb = np.reshape(yt,(m,1))

### -------------------------------------------------------------------------------------

### ---------------- SET VARIABLES (required) -------------------------------------------

x = xb
y = yb
learning_rate = 0.01 #(alpha)
iterations = 1500
initial_theta = np.zeros(shape=(2,1))

### -------------------------------------------------------------------------------------

### ---------------- ALGORITHM ----------------------------------------------------------

m = y.size

#plot data
fig, ax = plt.subplots()
ax.scatter(xt, yt, vmin=0, vmax=100)
plt.show(block=False)

#cost function
def costFunction(theta):
    error = np.subtract(np.matmul(x,theta),y)
    J = (1/(2*m))*(error*error).sum()
    return J

#gradient descent
def gradientDescent(theta,alpha,num_iters):
    J_history = np.zeros(shape=(1500))
    for i in range(1,num_iters+1):
        theta = theta -(1/m)*alpha*( x.T.dot(np.dot(x,theta) - y))
        J_history[i-1] = costFunction(theta)
    return theta, J_history

theta, J_history = gradientDescent(initial_theta,learning_rate,iterations)

#plot new function
x2 = np.linspace(0,25,100)
y2 = theta[1,0]*x2 + theta[0,0]
ax.plot(x2,y2)

print('Final Hypothesis: '+str(theta[1,0])+'x + '+str(theta[0,0]))

plt.show(block = True)

### --------------------------------------------------------------------------------------

