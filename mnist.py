from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[1], cmap='gray')

x_train = x_train.reshape((x_train.shape[0], 28*28)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype('float32')
x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add(Dense(32, input_dim=28*28, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=128)

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print(f"Test accuracy: {acc}")

y_pred = model.predict(x_test)

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
print(f"Predicted label: {np.argmax(y_pred[0], axis=-1)}")
print(f"True label: {y_test[0]}")



