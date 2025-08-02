import pandas as pd

# Step 1: Loading the dataset
file_path = r"C:\Users\prave\OneDrive\Desktop\Project TATA IQ.Geldium\PythonEDA\Fully_Cleaned_Delenquency_Prediction_Dataset.xlsx"  
df = pd.read_excel(file_path)

# Step 2: Displaying dataset info
print("Dataset Info:")
print(df.info())

# Step 3: Checking missing values in each column
print("\n Missing Values by Column:")
print(df.isnull().sum())

df['Income_missing'] = df['Income'].isnull().astype(int)
print("Rows with missing Income:")
print(df[df["Income_missing"] == 1])

# Example: missing income vs Employment_Status
print(df.columns.tolist())
print(df.groupby('Employment_Status')['Income_missing'].mean())
print(df.groupby('Credit_Card_Type')['Income_missing'])

print("Remaining missing 'Employed' incomes:", df[(df['Employment_Status'] == 'EMP') & (df['Income'].isnull())].shape[0])
