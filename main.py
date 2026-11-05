import pandas as pd
df = pd.read_csv("student_scores.csv")
#print(df)

#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.describe())

#Check for missing values and duplicates
#print("Missing values:")
#print(df.isnull().sum())
#print("Duplicate rows:")
#print(df.duplicated().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Separate input and output
X = df[["Hours"]]
y = df["Scores"]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#print("Training data:", X_train.shape)
#print("Testing data:", X_test.shape)

#train and make predeictions
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#print("Actual scores:", list(y_test))
#print("Predicted scores:", y_pred)

#evaluate and test model
#print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
#print("R2 Score:", r2_score(y_test, y_pred))
new_prediction = model.predict(pd.DataFrame({"Hours": [7]}))
#print("Predicted score for 7 study hours:", new_prediction[0])

import matplotlib.pyplot as plt
plt.scatter(X, y, color="blue", label="Actual scores")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.title("Student Score Prediction")
plt.legend()
plt.show()



