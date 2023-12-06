import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.dummy import DummyRegressor  #Use DummyRegressor for regression problems
from sklearn.metrics import mean_squared_error

df = pd.read_csv('CombinedPrediction/filtered_data_Combined_decimal.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df['utility_agent1']

#Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#Initialize the Naive model for regression
naive_model = DummyRegressor(strategy='mean')

"""
#Train the naive model (fitting is not needed for DummyRegressor)

#Make predictions on the validation set
y_pred_naive = naive_model.predict(X_val)

#Evaluate the naive model
mse_naive = mean_squared_error(y_val, y_pred_naive)
print(f'Mean Squared Error (Naive Model): {mse_naive}')
"""

#Set up k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True)

#Make predictions using cross-validation
y_pred_naive = cross_val_predict(naive_model, X, y, cv=kfold)

#Evaluate the naive model using mean squared error
mse_naive = mean_squared_error(y, y_pred_naive)
print(f'MSE (Naive Model): {mse_naive}')
