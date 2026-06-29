import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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