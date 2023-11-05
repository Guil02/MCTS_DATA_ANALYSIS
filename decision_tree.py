import polars as pl

df = pl.read_csv('decision_tree_csv/expansion_data.csv')
df = df.filter((df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS'))
df = df.drop(['agent1_AI_type', 'agent2_AI_type'])
print(df)

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
df.write_csv('decision_tree_csv/expansion_data_grouped.csv')
