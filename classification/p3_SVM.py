import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score

df = pd.read_csv('training_data/sets/components classification.csv')

#Convert TRUE and FALSE to 1 and 0
df.replace({True: 1, False: 0}, inplace=True)

#Remove rows with Random
df = df[(df['Selection 1 - Random'] != 1) & (df['Selection 2 - Random'] != 1)]

#Encode categorical variables
label_encoder = LabelEncoder()
df['Most common outcome for Agent 1'] = label_encoder.fit_transform(df['Most common outcome for Agent 1'])

X = df.drop('Most common outcome for Agent 1', axis=1)
y = df['Most common outcome for Agent 1']

#Use StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Create an SVM classifier
svm_classifier = SVC(random_state=42)

#Perform 5-fold cross-validation
f1_scorer = make_scorer(f1_score, average='macro')
cross_val_results_accuracy = cross_val_score(svm_classifier, X, y, cv=cv, scoring='accuracy')
cross_val_results_f1 = cross_val_score(svm_classifier, X, y, cv=cv, scoring=f1_scorer)

#Print results
print(f"Cross-validation Accuracy: {cross_val_results_accuracy.mean():.2f}")
print(f"Cross-validation F1 Score: {cross_val_results_f1.mean():.2f}")
