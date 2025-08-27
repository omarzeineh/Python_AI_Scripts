import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
clean_data_C = pd.read_csv("cleaned_merged_data.csv")

features = clean_data_C.drop(['MET', 'C', 'ANALYSIS_TIMESTAMP'], axis=1).values
C = clean_data_C['C'].values
MET = clean_data_C['MET'].values

features_train, features_test, C_train, C_test, MET_train, MET_test = train_test_split(features, C, MET, test_size=0.2)



C_model = RandomForestRegressor()
C_model.fit(features_train, C_train)

MET_model = RandomForestRegressor()
MET_model.fit(features_train, MET_train)

C_pred = C_model.predict(features_train)
MET_pred = MET_model.predict(features_train)

C_rmse = root_mean_squared_error(C_train, C_pred)
MET_rmse = root_mean_squared_error(MET_train, MET_pred)

print('C_rmse', C_rmse)

print('MET_rmse', MET_rmse)
