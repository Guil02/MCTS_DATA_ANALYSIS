import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# Load your dataset
df = pd.read_csv('new_csv/filtered_file.csv')

# Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target values (y)
X = df.drop(['utility_agent1', 'utility_agent2'], axis=1)
y = df[['utility_agent1', 'utility_agent2']]

# Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with 0 in training data
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train_imputed = imputer.fit_transform(X_train)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on imputed data
model.fit(X_train_imputed, y_train)

# Impute missing values with 0 in validation data
X_val_imputed = imputer.transform(X_val)

# Make predictions on the imputed validation set
y_pred = model.predict(X_val_imputed)

# Define a custom rounding function
def custom_round(x, deviation=0.1):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0

# Apply the custom rounding function to predictions
v_custom_round = np.vectorize(custom_round)
y_pred = v_custom_round(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_val.values.flatten(), y_pred.flatten())
print(f'Accuracy: {accuracy}')
