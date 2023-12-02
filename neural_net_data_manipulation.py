import pandas as pd
# from pathlib import Path
#
# df = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None)
# df = df.filter(df['NumPlayers'] == 2.0)
# df = df.select(['GameRulesetName', 'Id',
#                 'agent1_AI_type', 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
#                 'agent2_AI_type', 'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out',
#                 'utility_agent1', 'utility_agent2'])
#
# if not Path.is_dir(Path('neural_net_csv')):
#     Path.mkdir(Path('neural_net_csv'))
#
# df.write_csv('neural_net_csv/base_data.csv')

df = pd.read_csv('neural_net_csv/base_data.csv')
df = df.loc[(df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS')]
df = df.drop(columns=['agent1_AI_type', 'agent2_AI_type'])

df_grouped = pd.pivot_table(df, values='utility_agent2', index=['GameRulesetName', 'Id',
                 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
                 'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'],
                            columns = ['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

df_grouped = df_grouped.reset_index()
df_grouped = df_grouped.rename(columns={-1.0: 'Losses', 0.0: 'Draws', 1.0: 'Wins'})
df_grouped = df_grouped.rename_axis(None, axis=1)

print(df_grouped)

df_grouped.to_csv('neural_net_csv/grouped_data.csv')
