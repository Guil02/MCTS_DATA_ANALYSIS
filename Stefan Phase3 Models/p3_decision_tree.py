import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('training_data\sets\components classification.csv')

#Convert TRUE and FALSE to 1 and 0
df.replace({True: 1, False: 0}, inplace=True)

df = df[(df['Selection 1'] != 'Random') & (df['Selection 2'] != 'Random')]

#Encode categorical variables
df = pd.get_dummies(df, columns=['Selection 1', 'Selection 2','Play-out 1','Play-out 2'])
label_encoder = LabelEncoder()
df['Most common outcome for Agent 1'] = label_encoder.fit_transform(df['Most common outcome for Agent 1'])

# Split the data into features (X) and target variable (y)
X = df.drop('Most common outcome for Agent 1', axis=1)
y = df['Most common outcome for Agent 1']

#Use StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = DecisionTreeClassifier(random_state=42)

#Perform 5-fold cross-validation
cross_val_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

#Print cross-validation results
print(f"Cross-validation Accuracy: {cross_val_results.mean():.2f}")