import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import time
import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(file_dir, "..")

sys.path.append(parent_dir)
from training_data.import_training_set import import_training_set

def regret_scores(true_labels, pred_labels):
    expected_value = (true_labels * pred_labels + (1 - true_labels) * (1 - pred_labels))
    regret = np.where(true_labels > 0.5, true_labels - expected_value, 1 - true_labels - expected_value)
    
    return regret

#Load data,weights and rulesetIDs
df, weights, rulesetIds = import_training_set("components classification")
exact_outcomes_set = pd.read_csv("training_data/sets/components regression.csv")

#Create a boolean mask to identify rows to keep
mask = (df['Selection 1 - Random'] != True) & (df['Selection 2 - Random'] != True)

#Apply the mask to the DataFrame
df = df[mask]

#Apply the mask to weights and rulesetIds
weights = weights[mask]
rulesetIds = rulesetIds[mask]

#Load exact outcomes for regret measurement
exact_outcomes = exact_outcomes_set["Win rate of Agent 1"][mask] 

X = df.drop(columns=["Most common outcome for Agent 1"])
y = df["Most common outcome for Agent 1"]

#Perform LOO
scores = []
rulesetWeights = []
regret_scores_list = []
noIterations = 0
noRulesets = rulesetIds.unique().shape[0]

startTime = time.time()
for rulesetId in rulesetIds.unique():
    trainIdx = (rulesetIds != rulesetId)
    X_train = X[trainIdx]
    y_train = y[trainIdx]

    testIdx = (rulesetIds == rulesetId)
    X_test = X[testIdx]
    y_test = y[testIdx]

    #Train model
    model = RandomForestClassifier(n_jobs=-1).fit(X_train, y_train, sample_weight=weights[trainIdx])

    #Predict on test set
    y_pred = model.predict(X_test)

    #Determine performance of classifier
    scores.append(f1_score(y_test, y_pred, average="weighted"))

    y_true_values = exact_outcomes[testIdx]
    y_pred_converted = np.array([{'Draw': 0.5, 'Loss': 0, 'Win': 1}[val] for val in y_pred])
    regret = regret_scores(y_true_values, y_pred_converted)
    regret_scores_list.append(regret.mean())

    #Log info
    rulesetWeights.append(weights[testIdx].iloc[0])

    #Output progress
    noIterations += 1
    if noIterations % 50 == 0:
        elapsedTime = time.time() - startTime
        avgTimePerIteration = elapsedTime / noIterations
        print(f"Progress: {noIterations}/{noRulesets}; Time elapsed (s): {elapsedTime:.2f}; Estimated time left (s): {(noRulesets - noIterations) * avgTimePerIteration:.2f}")

print(f"Progress: {noIterations}/{noRulesets}; Time elapsed (s): {time.time() - startTime:.2f}")

#Output results
print(f"Average regret: {np.mean(regret_scores_list):.2f}")
print(f"Minimum: {np.min(scores):.3f}/nMaximum: {np.max(scores):.3f}")
print(f"/nUnweighted average: {np.average(scores):.3f}/nWeighted average: {np.average(scores, weights=rulesetWeights):.3f}")
print(f"/nStd: {np.std(scores):.3f}")