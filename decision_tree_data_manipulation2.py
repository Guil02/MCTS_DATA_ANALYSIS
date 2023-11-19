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

#Identify the columns to group by by excluding 'utility_agent1' and 'utility_agent2'
#group_columns = [col for col in df.columns if col not in ['utility_agent1', 'utility_agent2']]

#Group by the similar columns and average 'utility_agent1' and 'utility_agent2' for those rows
#df = df.groupby(group_columns, as_index=False)[['utility_agent1', 'utility_agent2']].mean()

#Remove rows that dont have the same Exploration and play out values for both their agents
condition = (df['agent1_Play-out'] == df['agent2_Play-out']) & (df['agent1_Exploration'] == df['agent2_Exploration'])
df = df[condition]

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

#List of columns to drop
columns_to_drop = ["GameRulesetName", "Id"]

#Drop the specified columns
df = df.drop(columns=columns_to_drop)
'''
important_columns = [
    'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
    'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out',
    'utility_agent1', 'utility_agent2'
]

#Group by all columns except 'utility_agent1' and 'utility_agent2'
grouped_df = df.groupby(df.columns.difference(['utility_agent1', 'utility_agent2']).tolist(), as_index=False)

#Calculate the average for 'utility_agent1' and 'utility_agent2'
result_df = grouped_df.agg({'utility_agent1': 'mean', 'utility_agent2': 'mean'})

#Reorder columns
result_df = result_df[important_columns + [col for col in result_df.columns if col not in important_columns]]

#Rename the new columns if needed
result_df = result_df.rename(columns={'utility_agent1': 'utility_agent1', 'utility_agent2': 'utility_agent2'})
'''

#Create copies of all rows with swapped positions for agents
new_rows = []
#Iterate through each row in the original DataFrame
for index, row in df.iterrows():
    #Create a copy of the row
    new_row = row.copy()

    #Swap values between agent1_Expansion and agent2_Expansion
    new_row['agent1_Expansion'], new_row['agent2_Expansion'] = new_row['agent2_Expansion'], new_row['agent1_Expansion']

    #Swap values between utility_agent1 and utility_agent2
    new_row['utility_agent1'], new_row['utility_agent2'] = new_row['utility_agent2'], new_row['utility_agent1']

    #Append the new row to the list
    new_rows.append(new_row)

#Concatenate the original DataFrame with the new rows
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

#Save the result DataFrame to a CSV file
df.to_csv("new_csv/filtered_file.csv", index=False)
