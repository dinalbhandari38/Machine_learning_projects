#for prediction, download a dataset and store it in same folder
#import important libraries
import pandas as pd #pandas for creating table
import numpy as np #numpy for numerical operations
import matplotlib.pyplot as plt #matplotlib for visualization

#read and create table of downloaded file
df=pd.read_csv("car_data.csv") #df is a table created using pandas
#print(df.head()) #show first 5 rows
#print(df.shape)  #show no of rows and columns
#print(df.info())  # show information such as min,max,mean,median etc
#print(df.columns) #show column names in the dataset

#feature and target
x=df.drop("selling_price",axis=1)   #input information for prediction
y=df["selling_price"]   #result needed 
#print("Features:")
#print(x.head())
#print("Target:")
#print(y.head())
print(x.select_dtypes(include="object").columns) #dtype of each columns

#split the data for training and testing
from sklearn.model_selection import train_test_split  
x = pd.get_dummies(x, drop_first=True) #columns get added to dataset where colomns have this or that options
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42) #test size divides data into 80 & 20
#print("Training data:",x_train.shape) #check rows and columns of training data
#print("Testing data:",x_test.shape)  #check rows and columns of testing data

#create and train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()  #use linreg for numerical data

#tarin
model.fit(x_train, y_train)

#prediction
prediction = model.predict(x_test)
#print("Actual values:")
#print(y_test.values[:10]) #print first 10 test values
#print("Predicted values:")
#print(prediction[:10]) ##print first 10 predicted values

#calculate mae, mse, r**2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, prediction) #MAE calculates the average absolute difference between actual and predicted values.
#Mean Squared Error (MSE)
mse = mean_squared_error(y_test, prediction) #MSE calculates the average of squared prediction errors.
#R² Score
r2 = r2_score(y_test, prediction) #R² measures how much of the variation in the target is explained by the model.
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#visualisation of actual vs predicted values
plt.figure(figsize=(8, 5)) #creates a graph.
plt.scatter(y_test, prediction) #plots actual values against predicted values.
plt.xlabel("Actual Values") #name the axes.
plt.ylabel("Predicted Values") #name the axes.
plt.title("Actual vs Predicted Values") #gives the graph a heading.
plt.show() #displays the graph.

plt.figure(figsize=(8, 5))
plt.scatter(y_test, prediction, label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect predictions"
) #creates the reference line for perfect predictions.
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend() #explains the reference line.
plt.show()


#check the model using a new product
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
#print("Predicted car price:", new_prediction[0])