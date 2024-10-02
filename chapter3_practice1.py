import numpy as np
x = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])
y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])
def logistic_func(z):
    return 1/(1+np.exp(-z))
def predict(x, Theta0, Theta1):
    z = Theta0 + Theta1 * x
    gz = logistic_func(z)
    return gz
def costfunc(x, y_true, Theta0, Theta1 ):
    m =len(x)
    epsilon = 1e-15
    y_pred = predict(x, Theta0, Theta1)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred + epsilon) + (1- y_true) * np.log(1- y_pred + epsilon))
    return cost
def gradient_descent(x, y, Theta0, Theta1, learning_rate):
    m=len(x)
    Gradient_Descent0 = Theta0 - learning_rate * (1/m) * np.sum(predict(x, Theta0, Theta1) - y)
    Gradient_Descent1 = Theta1 - learning_rate * (1/m) * np.sum((predict(x, Theta0, Theta1) - y)*x)
    Theta0 = Gradient_Descent0
    Theta1 = Gradient_Descent1
    return Theta0, Theta1
np.random.seed()
Theta0 = np.random.rand()
Theta1 = np.random.rand()
learning_rate = 1e-9
iteration = 1000
for i in range(0,iteration):
    Theta0, Theta1 = gradient_descent(x, y, Theta0, Theta1, learning_rate)
cos = costfunc(x, y, Theta0, Theta1)
print(F"{Theta0} + {Theta1}*x")
print(cos)