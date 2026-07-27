import pandas as pd
from sklearn.linear_model import LinearRegression

# for Employment_Status = EMP

#Loading the dataset
df = pd.read_excel(r"C:\Users\prave\OneDrive\Desktop\Project TATA IQ.Geldium\PythonEDA\EDACleaned_Delinquency_Dataset.xlsx")

employed_df = df[df["Employment_Status"] == "EMP" ]

# Splitting into missing and non-missing income
missing_income_employed = employed_df[employed_df["Income"].isnull()]
non_missing_income_employed = employed_df[employed_df["Income"].notnull()]

# Defining features to use
features = ["Age", "Loan_Balance", "Credit_Utilization", "Debt_to_Income_Ratio", "Account_Tenure"]

# Dropping rows with missing predictor values in training data
non_missing_income_employed = non_missing_income_employed.dropna(subset=features)

# Trainning data
X_train = non_missing_income_employed[features]
y_train = non_missing_income_employed["Income"]

# Testing data (only where features are complete)
X_test = missing_income_employed.dropna(subset=features)[features]
test_indices = X_test.index

# Trainning model and predict
model = LinearRegression()
model.fit(X_train, y_train)
predicted_income = model.predict(X_test).round(0).astype(int)

# Filling predicted values
df.loc[test_indices, "Income"] = predicted_income

# Printing updated rows
print(df.loc[test_indices, ["Employment_Status", "Income"]])
print("Remaining missing 'Employed' incomes:", df[(df['Employment_Status'] == 'EMP') & (df['Income'].isnull())].shape[0])
df.to_excel("EDACleaned_Delinquency_Dataset_.xlsx", index=False)
