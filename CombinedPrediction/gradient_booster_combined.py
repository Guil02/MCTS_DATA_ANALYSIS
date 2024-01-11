import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('CombinedPrediction/filtered_data_Combined_01.csv')

#Identify and one-hot encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Separate features (X) and target values (y)
X = df.drop(['utility_agent1'], axis=1)
y = df[['utility_agent1']]

#Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)


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
"""
#Evaluate the model
accuracy = accuracy_score(y_val.values.flatten(), y_pred.flatten())
print(f'Accuracy (Gradient Boosting): {accuracy}')

#Confusion Matrix for 3 classes
cm = confusion_matrix(y_val.values.flatten(), y_pred.flatten())
class_names = sorted(y['utility_agent1'].unique())

#Calculate the total number of actual instances for each class
total_actual_instances = cm.sum(axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            xticklabels=[f'Predicted {class_name}' for class_name in class_names],
            yticklabels=[f'Actual {class_name} (n={total_instances})' for class_name, total_instances in zip(class_names, total_actual_instances)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()