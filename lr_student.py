import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\StudentStudyHour.csv.xls")
# print(df)

y=df['Scores']
x=df.drop('Scores',axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()

x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
modle=lr.fit(x_train,y_train)
y_pred=modle.predict(x_test)
df_pred=pd.DataFrame(({'Actual':y_test,'Predicted':y_pred}))