# type:ignore
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd

df = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\and.csv")
print(df)

x=df.iloc[:,:2].values
y=df.iloc[:,2].values

model= Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(x,y,epochs=250)
weights=model.get_weights()
print(weights[1])
