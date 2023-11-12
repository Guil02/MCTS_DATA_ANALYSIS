import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("new_csv/output_dataset.csv")

# Filter rows to remove games with random agents
df = df[~(df['agent1_AI_type'].str.lower().str.contains('random') | df['agent2_AI_type'].str.lower().str.contains('random'))]

# Filter to only keep games with 2 players
df = df.loc[df['NumPlayers'] == 2.0]

# Find columns with only one unique value
single_value_columns = df.columns[df.nunique() == 1]

# Drop columns with only one unique value
df = df.drop(columns=single_value_columns)

# Identify the columns to group by by excluding 'utility_agent1' and 'utility_agent2'
group_columns = [col for col in df.columns if col not in ['utility_agent1', 'utility_agent2']]

# Group by the similar columns and average 'utility_agent1' and 'utility_agent2' for those rows
df = df.groupby(group_columns, as_index=False)[['utility_agent1', 'utility_agent2']].mean()



df.to_csv("filtered_file.csv", index=False)
