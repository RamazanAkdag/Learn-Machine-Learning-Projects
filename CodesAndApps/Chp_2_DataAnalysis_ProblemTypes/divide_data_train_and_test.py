# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:58:08 2024

@author: akdag
"""
# ders 6 : kutuphanelerin yüklenmesi
            
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
data = pd.read_csv('../../DatasAndTools/eksikveriler.csv')

print(data)
    



# veri önişleme
boy = data[['boy']]
print(boy)
boykilo = data[['boy','kilo']]
print(boykilo)


# eksik veriler 

# sci - kit learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')



#encoder
ulke = data.iloc[:,0:1].values
print(ulke)

#label encoding

"""ilk olarak ülke kolonu sayısal değere çevirilir ve sonra onu kategorik olacak şekilde encode etmemiz lazım"""
from sklearn import preprocessing

yas = data.iloc[:,1:4].values
print(yas)

imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])

print(yas)

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke) # sayısal değere çevrildi

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

#numpy dizileri dataframe a dönüştürme

sonuc  = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
print(sonuc)


sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = data.iloc[:,-1].values
print(cinsiyet)

#dataframe birleştirme

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin eğitim ve test için ayrılması 

# split data to train and test 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)










