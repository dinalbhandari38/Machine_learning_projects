import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

#load dataset
diabetes = load_diabetes()

df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)
df["target"] = diabetes.target
#print(df.head())

#print("Dataset shape:", df.shape)
#print("\nDataset information:")
#print(df.info())
#print("\nStatistical summary:")
#print(df.describe())

X = df.drop("target", axis=1)
y = df["target"]
#print("Features:")
#print(X.head())
#print("Target:")
#print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#print("Training data:", X_train.shape)
#print("Testing data:", X_test.shape)

#create and train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#make prediction
predictions = model.predict(X_test)
#print("Actual values:")
#print(y_test.values[:10])
#print("Predicted values:")
#print(predictions[:10])

#calculate mae mse r**2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#calculate metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
#print("Mean Absolute Error:", mae)
#print("Mean Squared Error:", mse)
#print("R2 Score:", r2)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, label="Predictions")
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