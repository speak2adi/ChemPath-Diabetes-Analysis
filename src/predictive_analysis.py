import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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

# For multiple regression
# y being the dependent variable(Target) and x the independent variable(Feature)
yy = df['Glucose'] #Series (1D)
xx = df[
    [
        'Pregnancies',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]
] #Dataframe 2D

#Splitting the data for multiple regression training
x_train, x_test, y_train, y_test = train_test_split(
    xx, yy,
    test_size = 0.2, random_state = 42
)

multiple_model = LinearRegression()
multiple_model.fit(x_train, y_train)

glucose_prediction = multiple_model.predict(x_test)
mul_mae = mean_absolute_error(y_test, glucose_prediction)
print(f"Mean Absolute Error:  {mul_mae:.2f} ")
mul_mse = mean_squared_error(y_test, glucose_prediction)
print(f"Mean Squared Error: {mul_mse:.2f} ")
mul_rmse = root_mean_squared_error(y_test, glucose_prediction)
print(f"Root Mean Squared Error: {mul_rmse:.2f} ")
mul_r2 = r2_score(y_test, glucose_prediction)
print(f"R2 Score: {mul_r2:.4f}")
print("Slope (Coefficient):", multiple_model.coef_)
print("Intercept:", multiple_model.intercept_)

xxx = df[['BMI']]
yyy = df['Glucose']

poly = PolynomialFeatures(
    degree = 2,
    include_bias = False
)

# Transforming the dataframe xxx so we can include another column with the 2nd degree polynomial
x_poly = poly.fit_transform(xxx)

#Important to note that Polynomial regression is just linear regression applied to transformed features
x_train, x_test, y_train, y_test = train_test_split(
    x_poly, yyy,
    test_size = 0.2, random_state = 42
)

polynomial_model = LinearRegression()
polynomial_model.fit(x_train, y_train)

glu_pred = polynomial_model.predict(x_test)
pol_mse = mean_squared_error(y_test, glu_pred)
print(f"Pol Mean Squared Error:  {pol_mse:.2f} ")
pol_mae = mean_absolute_error(y_test, glu_pred)
print(f"Pol Mean Absolute Error:  {pol_mae:.2f} ")
pol_rmse = root_mean_squared_error(y_test, glu_pred)
print(f"Pol Root Mean Squared Error:  {pol_rmse:.2f} ")
pol_r2 = r2_score(y_test, glu_pred)
print(f"Pol R2 Score: {pol_r2:.4f}")



