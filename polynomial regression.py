#import libraries
import numpy as np
import matplotlib.pyplot as plt

### -------------------- GENERATE DATA --------------------------------------------------

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
learning_rate = 0.001 #(alpha)
iterations = 7000

### -------------------------------------------------------------------------------------

### ---------------- ALGORITHM ----------------------------------------------------------

m = y.size

#plot data
fig, ax = plt.subplots()
ax.scatter(x_training, y, vmin=0, vmax=100)
plt.show(block=False)
#x = np.column_stack((np.ones(shape=(m)),x_training,x_training**2,x_training**3,x_training**4,x_training**5,x_training**6,x_training**7,x_training**8,x_training**9,x_training**10,x_training**11,x_training**12,x_training**13,x_training**14,x_training**15,x_training**16,x_training**17,x_training**18,x_training**19,x_training**20))
x = np.column_stack((np.ones(shape=(m)),x_training,x_training**2,x_training**3))
num_rows,num_cols = x.shape
initial_theta = np.zeros(shape=(num_cols,1))
print(initial_theta)

#cost function
def costFunction(theta):
    error = np.subtract(np.dot(x,theta),y)
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

theta,J_history = gradientDescent(initial_theta,learning_rate,iterations)

#plot new function
x2 = np.linspace(np.amin(x_training),np.amax(x_training),100)
#y2 = theta[0,0] + theta[1,0]*x2 + theta[2,0]*x2**2 + theta[3,0]*x2**3 + theta[4,0]*x2**4 + theta[5,0]*x2**5 + theta[6,0]*x2**6 + theta[7,0]*x2**7 + theta[8,0]*x2**8 + theta[9,0]*x2**9 + theta[10,0]*x2**10 + theta[11,0]*x2**11 + theta[12,0]*x2**12 + theta[13,0]*x2**13 + theta[14,0]*x2**14 + theta[15,0]*x2**15 + theta[16,0]*x2**16 + theta[17,0]*x2**17 + theta[18,0]*x2**18 + theta[19,0]*x2**19 + theta[20,0]*x2**20
print(theta)
y2 = theta[0,0] + theta[1,0]*x2 + theta[2,0]*x2**2 + theta[3,0]*x2**3
ax.plot(x2,y2)

print('Final Hypothesis: '+str(theta[3,0])+'x^3 + '+str(theta[2,0])+'x^2 + '+str(theta[1,0])+'x + '+str(theta[0,0]))

plt.show(block = True)

### --------------------------------------------------------------------------------------

