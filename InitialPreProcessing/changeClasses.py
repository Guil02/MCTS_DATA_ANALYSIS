import pandas as pd

input_csv = 'PlayoutPrediction/filtered_data_Playout.csv'
df = pd.read_csv(input_csv)

# Replace all -1 values in the class column with 0
df['utility_agent1'] = df['utility_agent1'].replace(-1, 0)

output_csv = 'PlayoutPrediction/filtered_data_Playout_1-1.csv'
df.to_csv(output_csv, index=False)

print(f"Processing completed!")
