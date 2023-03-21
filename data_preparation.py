import numpy as np
import pandas as pd
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
predictions = regr.predict(X_test)

print(regr.coef_)



