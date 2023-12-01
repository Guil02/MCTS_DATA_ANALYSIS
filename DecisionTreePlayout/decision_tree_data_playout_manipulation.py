import pandas as pd

#Load the CSV file into a DataFrame
df = pd.read_csv("new_csv/output_dataset.csv")

#Filter rows to remove games with random agents
df = df[~(df['agent1_AI_type'].str.lower().str.contains('random') | df['agent2_AI_type'].str.lower().str.contains(
    'random'))]

#Filter to only keep games with 2 players
df = df.loc[df['NumPlayers'] == 2.0]

#Find columns with only one unique value
single_value_columns = df.columns[df.nunique() == 1]

#Drop columns with only one unique value
df = df.drop(columns=single_value_columns)

#Remove rows that dont have the same Exploration and play out values for both their agents
#condition = (df['agent1_Play-out'] == df['agent2_Play-out']) & (df['agent1_Exploration'] == df['agent2_Exploration'])
#df = df[condition]

#Iterate through the DataFrame to identify and swap rows that are the same game but with swapped player 1 and player 2
for index, row in df.iterrows():
    #Check if the current row has a corresponding swapped row
    swapped_row = df[(df['GameRulesetName'] == row['GameRulesetName']) &
                     (df['agent1_Exploration'] == row['agent2_Exploration']) &
                     (df['agent1_Play-out'] == row['agent2_Play-out']) &
                     (df['agent2_Exploration'] == row['agent1_Exploration']) &
                     (df['agent2_Play-out'] == row['agent1_Play-out']) &
                     (df['agent1_Expansion'] == row['agent2_Expansion']) &
                     (df['agent2_Expansion'] == row['agent1_Expansion'])]

    #If a swapped row is found, swap the values of agent1_Expansion, agent2_Expansion, utility_agent1, and utility_agent2
    if not swapped_row.empty:
        df.at[index, 'agent1_Expansion'], df.at[index, 'agent2_Expansion'] = row['agent2_Expansion'], row['agent1_Expansion']
        df.at[index, 'utility_agent1'], df.at[index, 'utility_agent2'] = row['utility_agent2'], row['utility_agent1']

#Identify the columns to group by by excluding 'utility_agent1' and 'utility_agent2'
group_columns = [col for col in df.columns if col not in ['utility_agent1', 'utility_agent2']]

#Group by the similar columns and average 'utility_agent1' and 'utility_agent2' for those rows
df = df.groupby(group_columns, as_index=False)[['utility_agent1', 'utility_agent2']].mean()

#List of columns to drop
columns_to_drop = ["GameRulesetName", "Id","agent1_Exploration","agent1_Expansion","agent2_Exploration","agent2_Expansion", "utility_agent2"]  #Playout Only

#Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Set utility to 1 -1 or 0 depending on threshold (0.2|-0.2)
df['utility_agent1'] = df['utility_agent1'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

#Save the result DataFrame to a CSV file
df.to_csv("DecisionTreePlayout/filtered_data_Playout", index=False)
