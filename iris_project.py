#imoport libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

#load dataset
iris = load_iris()

#print dataset as table
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
#print(df.head())

#features and target
X = df.drop("species", axis=1)
y = df["species"]
#print("Features:")
#print(X.head())
#print("Target:")
#print(y.head())

#print(iris.target_names)

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#print("Training data:", X_train.shape)
#print("Testing data:", X_test.shape)

#create and train model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#make predictions
predictions = model.predict(X_test)
print("Actual values:")
print(y_test.values)
print("Predicted values:")
print(predictions)

#measure model accurracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

#confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
display.plot()
plt.show()

#visualise dataset
plt.figure(figsize=(8, 5))
for species in range(3):
    subset = df[df["species"] == species]
    plt.scatter(
        subset["sepal length (cm)"],
        subset["petal length (cm)"],
        label=iris.target_names[species]
    )
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Flower Classification Dataset")
plt.legend()
plt.show()

#test model with new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]
print("Predicted flower:", predicted_species)