
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("/content/gBqE3R1cmOb0qyAv.csv")

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nDataset Head:\n", df.head())

# Set pandas to display all columns
pd.set_option("display.max_columns", None)

# Drop customerID as it’s not a predictive feature
df = df.drop(columns=['customerID'])

# Handle missing values (if any, replace with 0 for simplicity as per dashboard logic)
df = df.fillna(0)

# Convert TotalCharges to numeric, handling errors
df['TotalCharges'] = df['TotalCharges'].replace({",": ""}, regex=True)
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['tenure', 'MonthlyCharges', 'TotalCharges'])

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['churned'])
y = df['churned']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save preprocessed data
X_resampled.to_csv("preprocessed_features.csv", index=False)
pd.Series(y_resampled, name='churned').to_csv("preprocessed_target.csv", index=False)

print("\nPreprocessed Data Shape:", X_resampled.shape)
print("Preprocessed Target Distribution:\n", pd.Series(y_resampled).value_counts())