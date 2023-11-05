import polars as pl
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = pl.read_csv('decision_tree_csv/X_expansion.csv')
y = pl.read_csv('decision_tree_csv/y_expansion.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
print(clf.score(X_test, y_test))
