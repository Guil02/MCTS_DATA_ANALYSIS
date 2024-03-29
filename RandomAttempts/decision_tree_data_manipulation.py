import polars as pl
from pathlib import Path

df = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None)
df = df.filter(df['NumPlayers'] == 2.0)
df = df.select(['GameRulesetName', 'Id', 'agent1_AI_type', 'agent1_Expansion', 'agent2_AI_type', 'agent2_Expansion',
                'utility_agent1', 'utility_agent2'])

if not Path.is_dir(Path('decision_tree_csv')):
    Path.mkdir(Path('decision_tree_csv'))

df.write_csv('decision_tree_csv/expansion_data.csv')

df = pl.read_csv('decision_tree_csv/expansion_data.csv')
df = df.filter((df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS'))
df = df.drop(['agent1_AI_type', 'agent2_AI_type'])

df_agent1 = df.select([
    pl.col('GameRulesetName'),
    pl.col('Id'),
    pl.col('agent1_Expansion').alias('Expansion'),
    pl.col('utility_agent1').alias('Utility')
])

df_agent2 = df.select([
    pl.col('GameRulesetName'),
    pl.col('Id'),
    pl.col('agent2_Expansion').alias('Expansion'),
    pl.col('utility_agent2').alias('Utility')
])

df = pl.concat([df_agent1, df_agent2])

df = df.group_by('GameRulesetName', 'Id', 'Expansion').agg(pl.sum('Utility').alias('Utility'))

max_utility = df.group_by('GameRulesetName').agg(pl.max('Utility').alias('MaxUtility'))
df_with_max_utility = df.join(
    max_utility,
    on='GameRulesetName',
    how='inner'
)

df_filtered = df_with_max_utility.filter(
    pl.col('Utility') == pl.col('MaxUtility')
)

df = df_filtered.select(['GameRulesetName', 'Id', 'Expansion', 'Utility'])

# print(df.shape)
# print(df.select(['Id']).unique())

df = df.with_columns(pl.col('GameRulesetName').is_duplicated().alias("is_duplicate"))

df = df.filter(pl.col("is_duplicate") == False)

# Drop the 'is_duplicate' column as it's no longer needed
df = df.drop("is_duplicate")

# print(df)

df.write_csv('decision_tree_csv/expansion_data_grouped.csv')

df = pl.read_csv('decision_tree_csv/expansion_data_grouped.csv')

df = df.with_columns([
    (df['Expansion'].str.replace('ProgressiveHistory', '0')
     .str.replace('UCB1Tuned', '1')
     .str.replace('UCB1GRAVE', '2')
     .str.replace('UCB1', '3'))
    .alias('Expansion_encoded'),
])

y = df.select(['Expansion_encoded'])

df2 = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None)
df2 = df2.drop(['agent1_AI_type', 'agent1_Expansion',
                'agent1_Exploration', 'agent1_Play-out',
                'agent2_AI_type', 'agent2_Expansion',
                'agent2_Exploration', 'agent2_Play-out',
                'utility_agent1', 'utility_agent2'])

df_temp = df.select(['Id'])
X = df_temp.join(df2, on='Id', how='left').unique()
X = X.drop(['Id', 'GameRulesetName'])

columns_to_drop = [col for col in X.columns if X[col].unique().shape[0] != 1]
X = X[columns_to_drop]

y.write_csv('decision_tree_csv/y_expansion.csv')
X.write_csv('decision_tree_csv/X_expansion.csv')
