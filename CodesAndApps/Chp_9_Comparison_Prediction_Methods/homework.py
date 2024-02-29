# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:56:47 2024

@author: akdag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm

datas = pd.read_csv('../../DatasAndTools/maaslar_yeni.csv')

x = datas.iloc[:,2:5]
y = datas.iloc[:,-1:]

X = x.values
Y = y.values



# data necessarity control
"""import statsmodels.api as sm


X = np.append(arr=np.ones((30,1)).astype(int), values=x, axis=1)

X_l = x.iloc[:,[0,1,2]].values
X_l = np.array(X_l,dtype=float)

model = sm.OLS(y,X_l).fit()
print(model.summary())"""



# Multiple Linear Regression

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

linear_regressor.fit(X, Y)

y_pred = linear_regressor.predict(X)



model = sm.OLS(y_pred,X).fit()
print("Linear Regression OLS results")
print(model.summary()) 


# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

model = sm.OLS(y_poly_pred,X).fit()
print()
print("Polynomial Regression OLS results")
print(model.summary()) 


# Scaling for SVM

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()

y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')

svr_reg.fit(x_olcekli, y_olcekli)

y_svr_pred = svr_reg.predict(x_olcekli)


model = sm.OLS(y_svr_pred,x_olcekli).fit()
print()
print("Support Vector Regression OLS results")
print(model.summary()) 


# Decision Tree

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X, Y)

y_dec_tree_pred = r_dt.predict(X)

model = sm.OLS(y_dec_tree_pred,X).fit()
print()
print("Decision Tree OLS results")
print(model.summary()) 



# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0, n_estimators=10)

rf_reg.fit(X, Y.ravel())

y_ra_fo_pred = rf_reg.predict(X)

model = sm.OLS(y_ra_fo_pred,X).fit()
print()
print("Random Forest OLS results")
print(model.summary()) 













