import pandas as pd 
import numpy as np
data=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\insurance_data.csv.xls")
print (data)

inputs=data.drop('bought_insurance',axis=1)
output=data.drop('age',axis=1)

import matplotlib.pyplot as plt
plt.xlabel("Age")
plt.ylabel("Bought insurance")
plt.scatter(inputs,output)

from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(inputs,output,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_trian)
y_pred=model.predict(x_test)
print(y_pred,x_test)
