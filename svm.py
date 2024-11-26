import pandas as pd
import numpy as np
path="C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\Social_Network_Ads.csv"
data=pd.read_csv(path)
print (data)

print(data['Gender'].unique())
print(data['Gender'].nunique())
data['Gender']=data['Gender'].map({'Male':'1', 'Female':'0'})

inputs=data.drop(['User ID','Purchased'],axis=1)
output= data['Purchased']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,output,train_size=0.8,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
model= SVC()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
print(data)
print(data.info)