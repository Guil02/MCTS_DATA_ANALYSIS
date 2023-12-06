import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('CombinedPrediction/filtered_data_Combined2.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df[['utility_agent1']]

#Split the data into training and validation sets (80/20 split)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)

"""
#Train the model (with the 80/20 split)
model.fit(X, y.values.flatten())

#Make predictions on the validation set
y_pred = model.predict(X_val)
"""

#Use k-fold validation
kfold = KFold(n_splits=10, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train the model
    model.fit(X_train, y_train.values.flatten())

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the model for each fold
    accuracy = accuracy_score(y_val.values.flatten(), y_pred.flatten())
    print(f'Accuracy (Gradient Boosting) - Fold {fold + 1}: {accuracy}')

#Evaluate the model
#accuracy = accuracy_score(y_val.values.flatten(), y_pred.flatten())
#print(f'Accuracy (Gradient Boosting): {accuracy}')
