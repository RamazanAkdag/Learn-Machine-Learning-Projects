# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:10:57 2024

@author: akdag
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


#veri yükleme
data = pd.read_csv('../../DatasAndTools/odev_tenis.csv')

print(data)


data2 = data.apply(LabelEncoder().fit_transform)

outlook = data2.iloc[:,0:1].values

print(outlook)

ohe = OneHotEncoder()

outlook = ohe.fit_transform(outlook).toarray()

print(outlook)

weather_data = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])
end_datas = pd.concat([weather_data, data.iloc[:,1:3]], axis=1)
end_datas = pd.concat([data2.iloc[:,3:5],end_datas],axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(end_datas.iloc[:,:-1],end_datas.iloc[:,-1:], test_size=0.40, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=end_datas.iloc[:,:-1], axis=1)

X_l = end_datas.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog=end_datas.iloc[:,-1:], exog=X_l)

r = r_ols.fit()
print(r.summary())

end_datas = end_datas.iloc[:,1:]

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=end_datas.iloc[:,:-1], axis=1)

X_l = end_datas.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog=end_datas.iloc[:,-1:], exog=X_l)

r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)












