import pandas as pd
from pathlib import Path

# df = pd.read_csv('new_csv/output_dataset.csv')
#
# df = df.loc[df['NumPlayers'] == 2.0]
# df = df.drop(columns=['NumPlayers'])
#
# df = df.loc[(df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS')]
# df = df.drop(columns=['agent1_AI_type', 'agent2_AI_type'])
#
# df = df[(df['agent1_Expansion'] != df['agent2_Expansion']) |
#         (df['agent1_Exploration'] != df['agent2_Exploration']) |
#         (df['agent1_Play-out'] != df['agent2_Play-out'])]
#
# df = df.drop(columns=df.columns[df.nunique() == 1])
#
# if not Path.is_dir(Path('neural_net_csv')):
#     Path.mkdir(Path('neural_net_csv'))
#
# df.to_csv('neural_net_csv/base_data.csv')

df = pd.read_csv('neural_net_csv/base_data.csv')

index_columns = df.columns[(df.columns != 'Unnamed: 0') & (df.columns != 'utility_agent1') & (df.columns != 'utility_agent2')]

print(index_columns.to_numpy())
print(['GameRulesetName', 'Id',
                 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
                 'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'])

# df_grouped = pd.pivot_table(df, values='utility_agent2', index=index_columns.to_numpy(),
#                             columns = ['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

# df_grouped = pd.pivot_table(df, values='utility_agent2', index=['GameRulesetName', 'Id',
#                  'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
#                  'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'],
#                             columns = ['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

df_grouped = df_grouped.reset_index()
df_grouped = df_grouped.rename_axis(None, axis=1)

df_grouped = df_grouped.rename(columns={-1.0: 'Losses', 0.0: 'Draws', 1.0: 'Wins'})
df_grouped['Games'] = df_grouped['Losses'] + df_grouped['Draws'] + df_grouped['Wins']
df_grouped['Win_Chance'] = (df_grouped['Wins'] + 0.5*df_grouped['Draws'])/df_grouped['Games']

print(df_grouped)

# print(df_grouped['Games'].value_counts().sort_index(ascending=False))

# df_grouped.to_csv('neural_net_csv/grouped_data.csv')
#
# df_grouped = pd.read_csv('neural_net_csv/grouped_data.csv')

# df_agent1_win_chance = df_grouped.groupby(['agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out'])\
#     .agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
# df_agent1_win_chance.columns = ['Expansion', 'Exploration', 'Play-out', 'agent1_Games', 'agent1_Chance']
#
# df_agent2_win_chance = df_grouped.groupby(['agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'])\
#     .agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
# df_agent2_win_chance.columns = ['Expansion', 'Exploration', 'Play-out', 'agent2_Games', 'agent2_Chance']
# df_agent2_win_chance['agent2_Chance'] = 1 - df_agent2_win_chance['agent2_Chance']
#
# df_win_chance = pd.merge(df_agent1_win_chance, df_agent2_win_chance, on=['Expansion', 'Exploration', 'Play-out'])
# df_win_chance['Total_Games'] = df_win_chance['agent1_Games'] + df_win_chance['agent2_Games']
# df_win_chance['Win_Chance'] = (df_win_chance['agent1_Games'] * df_win_chance['agent1_Chance'] + df_win_chance['agent2_Games'] * df_win_chance['agent2_Chance']) / df_win_chance['Total_Games']
# df_win_chance = df_win_chance.sort_values(by=['Win_Chance'])
#
# print(df_win_chance)
#
# df_player_one_bias = df_grouped.groupby(['GameRulesetName', 'Id']).agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
# df_player_one_bias.columns = df_player_one_bias.columns.get_level_values(0)
# df_player_one_bias = df_player_one_bias.sort_values(by=['Win_Chance'])
#
# print(df_player_one_bias)

