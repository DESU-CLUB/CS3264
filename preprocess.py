import pandas as pd
import csv
# Read the CSV file using semicolon as the separator.
df = pd.read_csv("evals/pristine_datasets/pima-diabetes.csv", sep=",")



# (Optional) If some values in smoking_history did not match the keys, they become NaN.
# You can fill them with a default value, for example:

missing_count = df.isnull().sum()  
missing_percent = 100 * df.isnull().sum() / len(df)
missing_report = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percent': missing_percent
})

print(missing_report)
# Check the DataFrame after processing.
print("\nProcessed DataFrame:")
print(df.head())
print("\nProcessed dtypes:")
print(df.dtypes)

# Save the processed DataFrame back to a CSV using semicolons.
df.to_csv("evals/dataset/pima-diabetes.csv",
          sep=";",
          index=False,
          quoting=csv.QUOTE_NONE,
          escapechar='\\')