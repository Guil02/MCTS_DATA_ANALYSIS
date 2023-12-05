import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_predict , KFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('CombinedPrediction/filtered_data_Combined.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df[['utility_agent1']]

#Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#Initialize the Naive model (always predicts the most frequent class)
naive_model = DummyClassifier(strategy='most_frequent')

"""
#Train the naive model (pick the most frequent class)
naive_model.fit(X_train, y_train)

#Make predictions on the validation set
y_pred_naive = naive_model.predict(X_val)

#Evaluate the naive model
accuracy_naive = accuracy_score(y_val.values.flatten(), y_pred_naive.flatten())
print(f'Accuracy (Naive Model): {accuracy_naive}')
"""

# Set up k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True)

# Make predictions using cross-validation
y_pred_naive = cross_val_predict(naive_model, X, y.values.flatten(), cv=kfold)

# Evaluate the naive model
accuracy_naive = accuracy_score(y.values.flatten(), y_pred_naive)
print(f'Average Accuracy (Naive Model): {accuracy_naive}')