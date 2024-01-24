import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import numpy as np

def save_feature_importance_to_excel(importance, names, filename):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'Feature Names': feature_names, 'Feature Importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['Feature Importance'], ascending=False, inplace=True)
    fi_df.to_excel(filename, index=False)

def calculate_regret(X, y, model, oracle_col):
    oracle_win_rate = df.groupby(oracle_col)['Most common outcome for Agent 1'].mean()
    predictions = model.fit(X, y).predict(X)
    
    #Ensure predictions have the same length as the original dataframe
    predictions = pd.Series(predictions, index=X.index)
    
    model_win_rate = df.groupby(predictions)['Most common outcome for Agent 1'].mean()
    regret = oracle_win_rate.max() - model_win_rate.max()
    return regret

df = pd.read_csv('training_data\sets\components classification.csv')

#Convert TRUE and FALSE to 1 and 0
df.replace({True: 1, False: 0}, inplace=True)

#Remove rows with Random
df = df[(df['Selection 1 - Random'] != 1) & (df['Selection 2 - Random'] != 1)]

df.to_csv("saved.csv",index=False)

#Encode categorical variables
label_encoder = LabelEncoder()
df['Most common outcome for Agent 1'] = label_encoder.fit_transform(df['Most common outcome for Agent 1'])

X = df.drop('Most common outcome for Agent 1', axis=1)
y = df['Most common outcome for Agent 1']

#Use StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Initialize Decision Tree model
model = DecisionTreeClassifier(random_state=42)

#Perform 5-fold cross-validation
f1_scorer = make_scorer(f1_score, average='macro')
cross_val_results_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cross_val_results_f1 = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer)

#Display results
print(f"Cross-validation Accuracy: {cross_val_results_accuracy.mean():.2f}")
print(f"Cross-validation F1 Score: {cross_val_results_f1.mean():.2f}")

#Save feature importance to Excel
model.fit(X, y)
save_feature_importance_to_excel(model.feature_importances_, X.columns, 'Stefan/Phase3 Models/feature_importance_decision_tree.xlsx')
