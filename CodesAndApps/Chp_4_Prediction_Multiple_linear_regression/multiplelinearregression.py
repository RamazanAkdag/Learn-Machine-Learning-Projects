# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:43:40 2024

@author: akdag
"""

# ders 6 : kutuphanelerin yüklenmesi
            
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
data = pd.read_csv('../../DatasAndTools/veriler.csv')

print(data)
    



# veri önişleme


# eksik veriler 

# sci - kit learn

ulke = data.iloc[:,0:1].values
print(ulke)

#label encoding

"""ilk olarak ülke kolonu sayısal değere çevirilir ve sonra onu kategorik olacak şekilde encode etmemiz lazım"""
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke) # sayısal değere çevrildi

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

ulke_sonuc  = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])




cinsiyet = data.iloc[:,4:5].values
print(cinsiyet)

cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])

print(cinsiyet)

cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

cinsiyet_sonuc = pd.DataFrame(data=cinsiyet[:,0:1], index=range(22), columns=['cinsiyet'])
print(cinsiyet_sonuc)


boy_kilo_yas = data.iloc[:,1:4].values
print(boy_kilo_yas)

bky_sonuc = pd.DataFrame(data=boy_kilo_yas, index=range(22), columns=['boy','kilo','yas'])


ulke_boy_kilo = pd.concat([ulke_sonuc, bky_sonuc], axis=1)

process_data = pd.concat([ulke_boy_kilo, cinsiyet_sonuc],axis=1)


# split data to train and test 
"""from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ulke_boy_kilo,cinsiyet_sonuc,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)"""


boy = process_data.iloc[:,3:4].values
boy = pd.DataFrame(data=boy, index=range(22), columns=['boy']) 

boy_sonrasi = process_data.iloc[:,4:].values
boy_sonrasi = pd.DataFrame(data=boy_sonrasi, index=range(22), columns=['kilo','yas','cinsiyet'])

yeni_process_data = pd.concat([ulke_sonuc, boy_sonrasi], axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(yeni_process_data,boy, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()

regressor2.fit(x_train, y_train)

y_pred = regressor2.predict(x_test)

# necessary and unnecesary variables 
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=yeni_process_data, axis=1)

X_l = yeni_process_data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(boy,X_l).fit()
print(model.summary())

X_l = yeni_process_data.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(boy,X_l).fit()
print(model.summary())


X_l = yeni_process_data.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)

model = sm.OLS(boy,X_l).fit()
print(model.summary())















