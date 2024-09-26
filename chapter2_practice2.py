import numpy as np
import math 
import pandas as pd
data = pd.read_csv('Practic2_Chapter2.csv')
x1 = data['TV']
x2 = data['Radio']
x3 = data['Newspaper']
y = data['Sales']
def predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3):
    return Theta0 + Theta1 * x1 + Theta2 * x2 + Theta3 * x3
def costfunc(x1, x2, x3, Theta0, Theta1, Theta2, Theta3,y):
    m = len(x1)
    return (1/2*m) * np.sum(( predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3)-y)**2)
def Gradient_Descent(x1, x2, x3, Theta0, Theta1, Theta2, Theta3, y, learning_rate):
    m= len(x1)
    Gradient_Descent0 = Theta0 - learning_rate * (1/m) * np.sum(predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3)-y)
    Gradient_Descent1 = Theta1 - learning_rate * (1/m) * np.sum((predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3)-y)*x1)
    Gradient_Descent2 = Theta2 - learning_rate * (1/m) * np.sum((predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3)-y)*x2)
    Gradient_Descent3 = Theta3 - learning_rate * (1/m) * np.sum((predict(x1, x2, x3, Theta0, Theta1, Theta2, Theta3)-y)*x3)
    Theta0 = Gradient_Descent0 
    Theta1 = Gradient_Descent1
    Theta2 = Gradient_Descent2
    Theta3 = Gradient_Descent3
    return Theta0, Theta1, Theta2, Theta3
Theta0 = 50
Theta1 = 0.1
Theta2 = 50
Theta3 = 0.75
learning_rate = 1e-6
for i in range (0,200):
    (Theta0, Theta1, Theta2, Theta3)= Gradient_Descent( x1, x2, x3, Theta0, Theta1, Theta2, Theta3, y, learning_rate)
print(f"{Theta0} + {Theta1}*x1 + {Theta2}*x2 + {Theta3}*x3")
