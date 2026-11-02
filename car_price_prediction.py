import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("car_data.csv")
#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.columns)


x=df.drop("selling_price",axis=1)
y=df["selling_price"]
#print("Features:")
#print(x.head())
#print("Target:")
#print(y.head())
print(x.select_dtypes(include="object").columns)

from sklearn.model_selection import train_test_split
x = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
#print("Training data:",x_train.shape)
#print("Testing data:",x_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)
prediction = model.predict(x_test)
#print("Actual values:")
#print(y_test.values[:10])
#print("Predicted values:")
#print(prediction[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, prediction)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, prediction, label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect predictions"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()



new_car = pd.DataFrame([{
    "name": "Maruti Swift Dzire VDI",
    "year": 2018,
    "km_driven": 50000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner"
}])
new_car_encoded = pd.get_dummies(new_car)
new_car_encoded = new_car_encoded.reindex(
    columns=x_train.columns,
    fill_value=0
)
new_prediction = model.predict(new_car_encoded)
print("Predicted car price:", new_prediction[0])