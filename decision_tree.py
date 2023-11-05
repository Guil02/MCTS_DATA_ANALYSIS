import polars as pl
from sklearn import tree

X = pl.read_csv('decision_tree_csv/X_expansion.csv')
y = pl.read_csv('decision_tree_csv/y_expansion.csv')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)
