import numpy as np
x1 = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])
y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])
x2 = x1**2
lambda_reg= 0.5
def logistic_func(z):
    return 1/(1+np.exp(-z))
def predict(x1, x2, Theta0, Theta1, Theta2):
    z = Theta0 + Theta1 * x1 + Theta2 * x2
    gz = logistic_func(z)
    return gz
def costfunc(x1, x2, y_true, Theta0, Theta1,Theta2 ):
    m =len(x1)
    epsilon = 1e-15
    y_pred = predict(x1, x2, Theta0, Theta1, Theta2)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred + epsilon) + (1- y_true) * np.log(1- y_pred + epsilon))
    reg_term = (lambda_reg / 2*m) * (Theta1**2 + Theta2**2)
    cost += reg_term
    return cost
def gradient_descent(x1, x2, y, Theta0, Theta1, Theta2, learning_rate):
    m=len(x1)
    Gradient_Descent0 = Theta0 - learning_rate * (1/m) * np.sum(predict(x1, x2, Theta0, Theta1, Theta2) - y)
    Gradient_Descent1 = Theta1 - learning_rate * (1/m) * (np.sum((predict(x1, x2, Theta0, Theta1, Theta2) - y)*x1)+ learning_rate*(lambda_reg/m)*Theta1)
    Gradient_Descent2 = Theta2 - learning_rate * (1/m) * (np.sum((predict(x1, x2, Theta0, Theta1, Theta2) - y)*x2)- learning_rate*(lambda_reg/m)*Theta2)
    Theta0 = Gradient_Descent0
    Theta1 = Gradient_Descent1
    Theta2= Gradient_Descent2
    return Theta0, Theta1, Theta2
np.random.seed()
Theta0 = np.random.rand()
Theta1 = np.random.rand()
Theta2 = np.random.rand()
learning_rate = 1e-9
iteration = 1000
for i in range(0,iteration):
    Theta0, Theta1, Theta2 = gradient_descent(x1, x2, y, Theta0, Theta1, Theta2, learning_rate)
cos = costfunc(x1, x2,  y, Theta0, Theta1, Theta2)
print(f"Theta0: {Theta0}, Theta1: {Theta1}, Theta2:{Theta2} và costfuntion: {cos}")