import pandas as pd
import numpy as np

df = pd.read_csv("imports-85.data", header=None, na_values="?")
#print(df.head())
#print("Shape:", df.shape)
#print(df.info())

columns = [
    "symboling", "normalized-losses", "make", "fuel-type",
    "aspiration", "num-of-doors", "body-style", "drive-wheels",
    "engine-location", "wheel-base", "length", "width",
    "height", "curb-weight", "engine-type", "num-of-cylinders",
    "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]
df.columns = columns
#print(df.head())
#print(df.isnull().sum())

missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100
#print(missing_percentage[missing_percentage > 0])

#print(df["normalized-losses"].describe())
#print("Median:", df["normalized-losses"].median())
#print("Mean:", df["normalized-losses"].mean())

import matplotlib.pyplot as plt
df["normalized-losses"].hist()
#plt.show()

df["normalized-losses"] = df["normalized-losses"].fillna(
    df["normalized-losses"].median()
)
#print(df["normalized-losses"].isnull().sum())

#print(df["num-of-doors"].value_counts())
#print("Mode:", df["num-of-doors"].mode()[0])

df["num-of-doors"] = df["num-of-doors"].fillna(
    df["num-of-doors"].mode()[0]
)
#print(df["num-of-doors"].isnull().sum())

#print(df[["bore", "stroke", "horsepower", "peak-rpm"]].describe())
for col in ["bore", "stroke", "horsepower", "peak-rpm"]:
    df[col] = df[col].fillna(df[col].median())
#print(df[["bore", "stroke", "horsepower", "peak-rpm"]].isnull().sum())

df = df.dropna(subset=["price"])
#print(df["price"].isnull().sum())
#print(df.shape)
#print(df.isnull().sum())

#print(df.duplicated().sum())
#print(df.dtypes)

#exploratory data analysis
categorical_cols = df.select_dtypes(include="str").columns
#for col in categorical_cols:
    #print("\n", col)
    #print(df[col].value_counts())

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
#for col in numerical_cols:
    #print("\n", col)
    #print(df[col].describe())

#figure out outliers
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["price"])
plt.title("Price Distribution - Boxplot")
plt.xlabel("Price")
#plt.show()

#remove outliers
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
#print("Q1:", Q1)
#print("Q3:", Q3)
#print("IQR:", IQR)
#print("Lower limit:", lower_limit)
#print("Upper limit:", upper_limit)

#actaul ouliers
outliers = df[df["price"] > upper_limit]
#print(outliers[["make", "engine-size", "horsepower", "price"]])

correlation = df.select_dtypes(include=["int64", "float64"]).corr()
#print(correlation["price"].sort_values(ascending=False))

#visulize relationships
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
#plt.show()

categorical_cols = df.select_dtypes(include="str").columns
#print(categorical_cols)
#print("Total categorical columns:", len(categorical_cols))

encoded = pd.get_dummies(df["fuel-type"])
#print(encoded.head())

df = pd.get_dummies(df, columns=categorical_cols)
#print(df.head())
#print("New shape:", df.shape)

X = df.drop("price", axis=1)
y = df["price"]
#print("Features shape:", X.shape)
#print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#print("Training features:", X_train.shape)
#print("Testing features:", X_test.shape)
#print("Training target:", y_train.shape)
#print("Testing target:", y_test.shape)

#scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   #Learns the scaling values from training data and applies scaling
X_test_scaled = scaler.transform(X_test)    #Applies those same scaling values to test data
#print("Scaled training shape:", X_train_scaled.shape)
#print("Scaled testing shape:", X_test_scaled.shape)

#train first model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
#print("Model training completed!")

#predict
y_pred = model.predict(X_test_scaled)
#print(y_pred[:5])

#predict with actual price
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
#print(comparison.head(10))

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
#print("MAE:", mae)
#print("MSE:", mse)
#print("RMSE:", rmse)
#print("R2 Score:", r2)

#Understand the model's errors
comparison["Error"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
)
comparison["Absolute Error"] = comparison["Error"].abs()
#print(comparison.sort_values(by="Absolute Error",ascending=False).head(10))

comparison["Actual Price"].describe()
#print("Largest actual price in test data:", y_test.max())
# print("Smallest actual price in test data:", y_test.min())
# print("\nMean actual price:", y_test.mean())
# print("Mean absolute error:", mae)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
#plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
#plt.show()

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
#print("Decision Tree training completed!")
#print("First 5 predictions:", dt_pred[:5])

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_pred)
#print("Decision Tree MAE:", dt_mae)
# print("Decision Tree MSE:", dt_mse)
# print("Decision Tree RMSE:", dt_rmse)
# print("Decision Tree R2:", dt_r2)

#compare linreg and decesion tree 
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MAE": [mae, dt_mae],
    "MSE": [mse, dt_mse],
    "RMSE": [rmse, dt_rmse],
    "R2 Score": [r2, dt_r2]
})
#print(results)

#rand forest regression
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)


rf_pred = rf_model.predict(X_test_scaled)
print("First 5 predictions:", rf_pred[:5])
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)
#print("Random Forest MAE:", rf_mae)
#print("Random Forest MSE:", rf_mse)
#print("Random Forest RMSE:", rf_rmse)
#print("Random Forest R2:", rf_r2)

#compare
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [mae, dt_mae, rf_mae],
    "MSE": [mse, dt_mse, rf_mse],
    "RMSE": [rmse, dt_rmse, rf_rmse],
    "R2 Score": [r2, dt_r2, rf_r2]
})
print(results)








