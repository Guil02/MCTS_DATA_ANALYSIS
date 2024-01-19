import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('training_data\sets\components regression.csv')

#Convert TRUE and FALSE to 1 and 0
df.replace({True: 1, False: 0}, inplace=True)

#Filter out rows with Random
df = df[(df['Selection 1'] != 'Random') & (df['Selection 2'] != 'Random')]

#Encode categorical variables
df = pd.get_dummies(df, columns=['Selection 1', 'Selection 2', 'Play-out 1', 'Play-out 2'])

X = df.drop('Win rate of Agent 1', axis=1)
y = df['Win rate of Agent 1']

#Create a pipeline with preprocessing and GradientBoostingRegressor
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(random_state=42))])

#Use KFold for cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

#Perform 5-fold cross-validation
cross_val_results = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')

average_absolute_mse = np.mean(np.abs(cross_val_results))
print(f"Average Absolute Mean Squared Error: {average_absolute_mse}")
