#import libraries
import numpy as np
import matplotlib.pyplot as plt

### ---------------- DATA COLLECTION -----------------------------------

def generate_data_set():
    np.random.seed(0)
    x = 2 - 3 * np.random.normal(0, 1, 20)
    y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x, y

### -------------------------------------------------------------------------------------

### ---------------- SET VARIABLES (required) -------------------------------------------

x_training,y = generate_data_set()
learning_rate = 0.01 #(alpha)
iterations = 7000

### -------------------------------------------------------------------------------------

### ---------------- ALGORITHM ----------------------------------------------------------

m = y.size

#plot data
fig, ax = plt.subplots()
ax.scatter(x_training, y, vmin=0, vmax=100)
plt.show(block=False)

x = np.column_stack((np.ones(shape=(m)),x_training,x_training**2))
num_rows,num_cols = x.shape
initial_theta = np.zeros(shape=(num_cols,1))

#cost function
def costFunction(theta):
    error = np.subtract(np.matmul(x,theta),y)
    J = (1/(2*m))*(error*error).sum()
    return J

#gradient descent
def gradientDescent(theta,alpha,iterations):
    J_history = np.zeros(shape=(iterations))
    for i in range(1,iterations+1):
        theta = theta -(1/m)*alpha*( x.T.dot(np.dot(x,theta) - y))
        J_history[i-1] = costFunction(theta)
        #print(J_history[i-1])
    return theta, J_history

theta, J_history = gradientDescent(initial_theta,learning_rate,iterations)

print(theta)

#plot new function
x2 = np.linspace(np.amin(x_training),np.amax(x_training),100)
y2 = theta[2,0]*x2**2 + theta[1,0]*x2 + theta[0,0]
ax.plot(x2,y2)

print('Final Hypothesis: '+str(theta[2,0])+'x^2 + '+str(theta[1,0])+'x + '+str(theta[0,0]))

plt.show(block = True)

### --------------------------------------------------------------------------------------

