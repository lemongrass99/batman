import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn. preprocessing import LabelEncoder
from sklearn import metrics

data=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\Housing.csv")
print(data)

x=data.drop('price', axis=1)
y=data['price']

label_encoder=LabelEncoder()
for column in x.select_dtypes(include=['object']).columns:
    x[column]=label_encoder.fit_transform(x[column])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("Mean absolute error:",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squred Error:",metrics.mean_squared_error(y_test,y_pred))

