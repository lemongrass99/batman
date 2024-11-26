import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score 

data=load_iris()
print(data)


x=data.data
y=data.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,train_size=0.25)

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(x_train,y_train)

y_pred=dc.predict(x_test)
print("Train data accutacy:",accuracy_score(y_true=y_train,y_pred=dc.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true=y_test,y_pred=y_pred))
