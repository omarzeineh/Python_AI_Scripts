import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

x_train = x[0:10]
x_test = x[10:15]
y_train = y[0:10]
y_test = y[10:15]

XT = [np.power(x_train, 0),np.power(x_train, 1),np.power(x_train, 2)] #3x10
print(np.shape(XT))
X = np.transpose(XT) #10x3
print(np.shape(X))

XTX = np.matmul(XT, X)
XTXinv = np.linalg.inv(XTX)
XTXinvXT = np.matmul(XTXinv, XT)
W = np.matmul(XTXinvXT, y_train)
print("Weights: ", W)

testResults = []
for i in range(len(x_test)):
    xi = x_test[i]
    testResults.append(W[0]+W[1]*xi+W[2]*xi**2)

print("X test set: ", x_test)
print("Y predicted: ", testResults)
print("Actual Y: ", y_test)

error = np.subtract(y_test, testResults)
errorSquared = np.power(error, 2)
meanErrorSquared = (1/len(errorSquared))*np.sum(errorSquared)
RMSE = np.sqrt(meanErrorSquared)

print("RMSE: ", RMSE)
