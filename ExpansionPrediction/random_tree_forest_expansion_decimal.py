import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('ExpansionPrediction/filtered_data_Expansion_decimal.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df['utility_agent1']

#Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42)

#Train the model
model.fit(X_train, y_train.values.flatten())

#Make predictions on the validation set
y_pred = model.predict(X_val)

#Evaluate the model using mean squared error
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error (Random Forest): {mse}')