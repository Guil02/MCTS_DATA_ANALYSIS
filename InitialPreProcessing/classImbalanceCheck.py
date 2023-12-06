import pandas as pd

df = pd.read_csv('CombinedPrediction/filtered_data_Combined.csv')

class_distribution = df['utility_agent1'].value_counts()

print("Class Distribution:")
print(class_distribution)

class_proportions = df['utility_agent1'].value_counts(normalize=True)

print("\nClass Proportions:")
print(class_proportions)
