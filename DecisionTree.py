from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import polars as pl
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print('Reading in the data')
df = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None).to_pandas()
df.dropna(inplace=True)
Y = df['agent1_Expansion']
X = df.iloc[:, 12:]
X = X.select_dtypes(include='number')
# X = X.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=177013)

print('Starting to train Decision tree')
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# print("training Random Forest")
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# print("training NN")
# nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128), random_state=177013, verbose=True)
# nn.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)
# y_pred_rf = rf.predict(X_test)
# y_pred_SGD = nn.predict(X_test)

print(accuracy_score(y_test, y_pred_clf))
# print(accuracy_score(y_test, y_pred_rf))
# print(accuracy_score(y_test, y_pred_SGD))
value_count = df['agent1_Play-out'].value_counts()

sns.barplot(x=value_count.index, y=value_count.values)
plt.title('Play-out strategies')
plt.show()

new_df = df
# iterate over all the different games in Id
