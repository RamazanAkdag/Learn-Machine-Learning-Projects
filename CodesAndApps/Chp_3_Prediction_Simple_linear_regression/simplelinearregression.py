
# ders 6 : kutuphanelerin yüklenmesi
            
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


#veri yükleme
data = pd.read_csv('../../DatasAndTools/satislar.csv')





# veri önişleme
aylar = data[['Aylar']]

satislar = data[['Satislar']]





# split data to train and test 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

"""
#model inşası
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)


tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train, y_train, 'o')  # 'o' parametresi noktaların daire şeklinde gösterilmesini sağlar
plt.xlabel('X Ekseni Etiketi')
plt.ylabel('Y Ekseni Etiketi')
plt.title('Veri Görselleştirme')
plt.show()











