#Iris program
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

iris = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\Iris.csv")
print(iris.head())

x = iris.iloc[:, :4].values
y = iris.iloc[:, 4].values

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

scaler_object = MinMaxScaler()
scaler_object.fit(x_train)
scaled_x_train = scaler_object.transform(x_train)
scaled_x_test = scaler_object.transform(x_test)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(scaled_x_train, y_train, epochs=250)

y_pred = model.predict(scaled_x_test)
print(f"First test sample's input: {scaled_x_test[0]}")
print(f"First test sample's true label (one-hot): {y_test[0]}")
print(f"First test sample's predicted label (probabilities): {y_pred[0]}")

y_prediction = np.argmax(y_pred, axis=-1)
y_actual = np.argmax(y_test, axis=-1)

results = model.evaluate(scaled_x_test, y_test)
print(f'Final test set loss: {results[0]:.4f}')
print(f'Final test set accuracy: {results[1]:.4f}')

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.show()