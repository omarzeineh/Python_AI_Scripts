import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data.csv')

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
y_predicted = model.predict(poly_features)

rmse = root_mean_squared_error(y, y_predicted)
print('Root Mean Squared', rmse)

x_test = poly_features[:, 1]

for i in range(len(x_test)):
    for k in range(len(x_test)-1):
        if x_test[k] > x_test[k+1]:
            temp = x_test[k+1]
            x_test[k+1] = x_test[k]
            x_test[k] = temp
            temp = y[k + 1]
            y[k + 1] = y[k]
            y[k] = temp

plt.plot(x_test, y, '-o')
plt.show()



