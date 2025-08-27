import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:\\uni stuff\\2024-2025\\Spring 2024-2025\\AIRE310\\data.csv')

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

y_train = y[0:10]
y_test = y[10:15]

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(x.reshape(-1, 1))

x_train = poly_features[0:10]
x_test = poly_features[10:15]

model = LinearRegression()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

print('X test set', x_test[:, 1])
print('Y predicted', y_predicted)
print('Actual Y', y_test)
rmse = root_mean_squared_error(y_test, y_predicted)
print('Root Mean Squared', rmse)
print('Model Coefficients:', model.intercept_, model.coef_)

x_test = x_test[:, 1]

#Bubble sort to plot
for i in range(len(x_test)):
    for k in range(len(x_test)-1):
        if x_test[k] > x_test[k+1]:
            temp = x_test[k+1]
            x_test[k+1] = x_test[k]
            x_test[k] = temp
            temp = y_predicted[k + 1]
            y_predicted[k + 1] = y_predicted[k]
            y_predicted[k] = temp



plt.plot(x_test, y_predicted, '-o')
plt.xlabel('x test')
plt.ylabel('y predicted')
plt.show()
plt.figure(figsize=(10, 6))
x = np.array(range(-10000,10000))
y = model.predict(poly.fit_transform(x.reshape(-1, 1)))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()