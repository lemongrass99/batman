import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\homeprices.csv")
print(data)


inputs = data[['area']] 
outputs = data['price'] 

model = LinearRegression()
model.fit(inputs, outputs)


y_pred = model.predict([[2500]])  
print(f"Predicted price for area : {y_pred[0]:.2f}")

plt.scatter(inputs, outputs, color='blue', label='Actual Data')
plt.title('Linear Regression: Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()

plt.show()