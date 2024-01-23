import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

def preprocess_data(data):
    # Preprocess data to extract agent names and utility scores
    preprocessed_data = defaultdict(list)

    for _, row in data.iterrows():
        agents = row['agents'].strip("()").split(" / ")
        utilities = list(map(float, row['utilities'].split(';')))
        if len(agents) == 2 and len(utilities) == 2:
            matchup = tuple(sorted(agents))
            preprocessed_data[matchup].append(utilities)

    return preprocessed_data

def calculate_average_scores(matchup_data):
    scores = defaultdict(Counter)
    for utility1, utility2 in matchup_data:
        scores['agent1']['total_score'] += utility1
        scores['agent2']['total_score'] += utility2
        scores['agent1']['total_matches'] += 1
        scores['agent2']['total_matches'] += 1
    return scores

def predict_winner_with_average(scores):
    average_score1 = scores['agent1']['total_score'] / scores['agent1']['total_matches']
    average_score2 = scores['agent2']['total_score'] / scores['agent2']['total_matches']
    return 1 if average_score1 > average_score2 else -1 if average_score2 > average_score1 else 0


def leave_one_out_cross_validation(preprocessed_data):
    loo = LeaveOneOut()
    y_true = []
    y_pred = []

    for matchup, utilities in tqdm(preprocessed_data.items(), desc="Processing matchups"):
        for train_index, test_index in loo.split(utilities):
            train_data = [utilities[i] for i in train_index]
            test_data = utilities[test_index[0]]

            scores = calculate_average_scores(train_data)
            predicted_winner = predict_winner_with_average(scores)

            actual_winner = 1 if test_data[0] > test_data[1] else -1 if test_data[1] > test_data[0] else 0

            y_true.append(actual_winner)
            y_pred.append(predicted_winner)

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

    return accuracy, f1


# Load the CSV file
file_path = '../new_csv/AllGames.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Preprocess the data
preprocessed_data = preprocess_data(data)

# Perform leave-one-out cross-validation
loo_accuracy, loo_f1_score = leave_one_out_cross_validation(preprocessed_data)
print(f"Leave-One-Out Cross-Validation Accuracy: {loo_accuracy}")
print(f"Leave-One-Out Cross-Validation F1 Score: {loo_f1_score}")
