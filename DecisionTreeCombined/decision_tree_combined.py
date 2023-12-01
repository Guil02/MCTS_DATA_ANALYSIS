import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

#Load your dataset
df = pd.read_csv('DecisionTreeCombined/filtered_file_Combined.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df[['utility_agent1']]

#Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

#Train the model
model.fit(X_train, y_train)

#Make predictions on the validation set
y_pred = model.predict(X_val)

#Evaluate the model
accuracy = accuracy_score(y_val.values.flatten(), y_pred.flatten())
print(f'Accuracy: {accuracy}')
#Compute precision and recall
precision = precision_score(y_val.values.flatten(), y_pred.flatten(), average='weighted')

print(f'Precision: {precision}')

'''
#Print predictions from the validation set
total_rows = X_val.shape[0]

#Initialize counter for correct predictions
correct_predictions = 0

for sample_index in range(total_rows):
    sample_features = X_val.iloc[[sample_index]]
    sample_prediction = model.predict(sample_features)

    #Apply the custom rounding function to the sample prediction

    print(f'\nPrediction for sample index {sample_index} in the validation set:')
    print(f'Actual Values: {y_val.iloc[sample_index].values}')
    print(f'Predicted Values: {sample_prediction}')

    #Check if the prediction is correct
    if np.all(np.isclose(sample_prediction, y_val.iloc[sample_index].values, atol=1e-6)):
        correct_predictions += 1

#Print the total number of correct predictions
print(f'\nTotal Correct Predictions: {correct_predictions} out of {total_rows}')
'''