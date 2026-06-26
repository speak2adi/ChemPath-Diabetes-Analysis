import pandas as pd
import matplotlib.pyplot as plt

#LOADING DATA
df = pd.read_csv("../data/diabetes.csv")
print(df.head())

#INSPECTING AND DESCRIBING THE DATA
print(df.info())
print(df.describe()) #Basic Statistical Analysis

# DESCRIPTIVE STATISTICS
#INVESTIGATING GLUCOSE
print("Median Glucose: ", df["Glucose"].median())
print("Mean Glucose: ", df["Glucose"].mean())
print("Mode Glucose: ", df["Glucose"].mode())

#COMPARING DIABETIC AND NON-DIABETIC PATIENTS

#GROUPED ANALYSIS/STRATIFICATION(splitting a population into subgroups and comparing them.) OF THE MAIN DATASET
diabetic = df[df["Outcome"] == 1]
non_diabetic = df[df["Outcome"] == 0]

print("Number of diabetic patients:", len(diabetic))
print("Number of non-diabetic patients:", len(non_diabetic))

print("Mean glucose (diabetic):", diabetic["Glucose"].mean())
print("Mean glucose (non-diabetic):", non_diabetic["Glucose"].mean())

#GLUCOSE STANDARD DEVIATION AND VARIANCE
print("Glucose Standard Deviation: ", df["Glucose"].std())
print("Glucose Variance: ", df["Glucose"].var())

#DIABETIC VS NON-DIABETIC VARIABILITY
print("Diabetic Glucose SD: ", diabetic["Glucose"].std())
print("Non-Diabetic Glucose SD: ", non_diabetic["Glucose"].std())

#VISUALIZING THE DISTRIBUTION
plt.hist(df["Glucose"], bins=15)

plt.title("Distribution of Glucose Levels")
plt.xlabel("Glucose")
plt.ylabel("Number of Patients")

plt.show()

#DIAGNOSTIC ANALYSIS
#Correlation and Covariance between Glucose and BMI
print("Correlation Between Glucose and BMI: ", df["Glucose"].corr(df["BMI"]))
print("Covariance Between Glucose and BMI: ", df["Glucose"].cov(df["BMI"]))

#Correlation and Covariance between Glucose and Insulin
print("Correlation Between Glucose and Insulin: ", df["Glucose"].corr(df["Insulin"]))
print("Covariance Between Glucose and Insulin: ", df["Glucose"].cov(df["Insulin"]))

#Correlation and Covariance between Glucose and BMI in Diabetic Patients
print("Correlation Between Glucose and BMI in Diabetic Patients: ", diabetic["Glucose"].corr(diabetic["BMI"]))
print("Covariance Between Glucose and BMI Diabetic Patients: ", diabetic["Glucose"].cov(diabetic["BMI"]))

#Correlation and Covariance between Glucose and Insulin in Diabetic Patients
print("Correlation Between Glucose and Insulin in Diabetic Patients: ", diabetic["Glucose"].corr(diabetic["Insulin"]))
print("Covariance Between Glucose and Insulin Diabetic Patients: ", diabetic["Glucose"].cov(diabetic["Insulin"]))

# Doing a Scatter Plot Between Glucose vs BMI and Glucose vs Insulin
plt.scatter(df["Glucose"], df["BMI"])
plt.title("Glucose Levels vs BMI")
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.show()

#CONDITIONAL PROBABILITY
# Number of diabetics given high glucose P(Diabetes | High Glucose)
high_glucose = df[df["Glucose"] >= 126]

high_glucose_diabetic = high_glucose[
    high_glucose["Outcome"] == 1
]

probability = (
        len(high_glucose_diabetic) / len(high_glucose)
)
print("Probability of being diabetic, given high glucose: ", probability)

# BAYES THEOREM
#Prior Probability P(diabetes)
p_diabetes = len(df[df["Outcome"] == 1]) / len(df)
print("Probability of diabetes: ", p_diabetes)

#Likelihood P(High Glucose | Diabetes)
p_high_given_diabetes = (
    len(df[(df["Outcome"] == 1) & (df["Glucose"] >= 126)])
    /
    len(df[df["Outcome"] == 1])
)

print("Probability of high glucose given diabetes :", p_high_given_diabetes)

#Evidence P(High Glucose)
p_high_glucose = (
    len(df[df["Glucose"] >= 126])
    /
    len(df)
)

print(p_high_glucose)

#Applying Bayes'
bayes = (
    p_high_given_diabetes * p_diabetes
    /
    p_high_glucose
)

print("Probability of Diabetes, given High Glucose: ", bayes)