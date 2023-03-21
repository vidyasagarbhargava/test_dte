import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import mean_squared_error

df = pd.read_csv('some_file.csv')

X_train, y_train , X_test, y_test = some_func(df)

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
test_predict = regr.predict(X_test)

print(mean_squared_error(y_test, test_predict)

print(regr.coef_)
