import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('../data/diabetes.csv')
print(df.head())

# y being the dependent variable(Target) and x the independent variable(Feature)
y = df['Glucose'] #Series (1D)
x = df[['BMI']] #Dataframe (2D)

#Before jumping into Prediction, we have to first visualize the dataset.
plt.figure(figsize = (8,5))
plt.scatter(x,y)
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('BMI vs Glucose')
plt.show()


# Splitting the dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(
x, y,
    test_size = 0.2, random_state = 42
)

print("shape: ", x_train.shape)
print("shape: ", y_train.shape)

model = LinearRegression() #Creating an instance of the Linear regression object
#Here we're trying to get the line of best fit through the training data, by calculating the line that minimizes the
# total squared error.
model.fit(x_train, y_train)

# Getting what our model has learned, so we can apply to y = mx + b being the formular for linear regression
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Testing the model predicting the glucose values for BMI test data
glucose_pred = model.predict(x_test)
print(glucose_pred[:10])

# Comparing the predicted value, with the actual test values
comparison = x_test.copy()
comparison['Actual Glucose'] = y_test
comparison['Predicted Glucose'] = glucose_pred

print(comparison.head(10))

# Evaluating the Linear Regression Model
# MAE, MSE and RMSE are used to answer how wrong our model is
# Using MAE(Mean Absolute Error)
mae = mean_absolute_error(y_test, glucose_pred)
print("Mean Absolute Error: ", mae)

#Using MSE(Mean Squared Error)
mse = mean_squared_error(y_test, glucose_pred)
print("Mean Squared Error: ", mse)

#Using RMSE (Root Mean Squared Error)
rmse = root_mean_squared_error(y_test, glucose_pred)
print("Root Mean Squared Error: ", rmse)

#Using R2 which explains how much in variation in the target variable the model explains
r2 = r2_score(y_test, glucose_pred)
print("R2 Score: ", r2)