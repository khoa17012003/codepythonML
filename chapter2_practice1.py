import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
x= np.array([155,180,164,162,181,182,173,190,171,170,181,182,189,184,209,210])
y= np.array([51,52,54,53,55,59,61,59,63,76,64,66,69,72,70,80])

def predict(x,Theta0,Theta1):
    return Theta0 +Theta1 * x
def cosfun(x,y,Theta0,Theta1):
    m = len (x)
    return (1/2*m)*np.sum(( predict(x,Theta0,Theta1) - y )**2)
def Gradient_Descent(x,y,Theta0,Theta1,learning_rate):
    m=len(x)
    Gradient_Descent0 = Theta0 - learning_rate * (1/m) * np.sum(predict(x, Theta0, Theta1) - y)
    Gradient_Descent1 = Theta1 - learning_rate * (1/m) * np.sum((predict(x, Theta0, Theta1) - y)*x)
    Theta0 = Gradient_Descent0
    Theta1 = Gradient_Descent1
    return Theta0, Theta1
cost_finale = []
Theta0 = 0
Theta1 = 0.5
learning_rate = 1e-6
for i in range(0,3000):
    (Theta0, Theta1) = Gradient_Descent(x, y, Theta0, Theta1, learning_rate)
print(f"{Theta0}+{Theta1}x")
predict_y = predict(x,Theta0,Theta1)
plt.scatter(x, y , color='blue', label= 'data points')
plt.plot(x,predict_y, color= 'red', label = 'linear regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
input("Press Enter to close the plot...")
