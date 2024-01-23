import os
import pandas as pd

# How to use in your code?
# Simply import this function as "from training_data import import_training_set"
# Call this function with the name of the training set you want to load, e.g., "import_training_set("agents regression")"


def import_training_set(name):
    original_wd = os.getcwd()

    # Change working directory
    dirs = original_wd.split("\\")
    for i in range(len(dirs)):
        if dirs[i] == "MCTS_DATA_ANALYSIS":
            break
    training_wd = "\\".join(dirs[:i + 1]) + "\\training_data"

    os.chdir(training_wd)

    # Load training set
    data = pd.read_csv(f"sets/{name}.csv")

    # Correct data types:
    # - object -> category
    # - int64 -> int32
    for col in data.columns:
        if data[col].dtype == "object":
            # convert to category
            data[col] = data[col].astype("category")
        elif data[col].dtype == "int64":
            # convert to int32
            data[col] = data[col].astype("int")

    # Load training weights
    weights = pd.read_csv("weights.csv").values[:, 0]

    # Load training ruleset IDs
    rulesetIds = pd.read_csv("rulesetIds.csv").values[:, 0].astype("str")

    # Roll back working directory
    os.chdir(original_wd)

    return data, weights, rulesetIds
