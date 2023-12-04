import pandas as pd
from pathlib import Path

df = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None)
df = df.filter(df['NumPlayers'] == 2.0)
df = df.select(['GameRulesetName', 'Id',
                'agent1_AI_type', 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
                'agent2_AI_type', 'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out',
                'utility_agent1', 'utility_agent2'])

if not Path.is_dir(Path('neural_net_csv')):
    Path.mkdir(Path('neural_net_csv'))

df.write_csv('neural_net_csv/base_data.csv')

df = pd.read_csv('neural_net_csv/base_data.csv')
df = df.loc[(df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS')]
df = df.drop(columns=['agent1_AI_type', 'agent2_AI_type'])

df_grouped = pd.pivot_table(df, values='utility_agent2', index=['GameRulesetName', 'Id',
                 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
                 'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'],
                            columns = ['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

df_grouped = df_grouped.reset_index()
df_grouped = df_grouped.rename_axis(None, axis=1)

df_grouped = df_grouped.rename(columns={-1.0: 'Losses', 0.0: 'Draws', 1.0: 'Wins'})
df_grouped['Games'] = df_grouped['Losses'] + df_grouped['Draws'] + df_grouped['Wins']
df_grouped['Win_Chance'] = (df_grouped['Wins'] + 0.5*df_grouped['Draws'])/df_grouped['Games']

print(df_grouped['Games'].value_counts().sort_index(ascending=False))

df_grouped.to_csv('neural_net_csv/grouped_data.csv')

df_grouped = pd.read_csv('neural_net_csv/grouped_data.csv')

df_agent1_win_chance = df_grouped.groupby(['agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out'])\
    .agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
df_agent1_win_chance.columns = ['Expansion', 'Exploration', 'Play-out', 'agent1_Games', 'agent1_Chance']

df_agent2_win_chance = df_grouped.groupby(['agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'])\
    .agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
df_agent2_win_chance.columns = ['Expansion', 'Exploration', 'Play-out', 'agent2_Games', 'agent2_Chance']
df_agent2_win_chance['agent2_Chance'] = 1 - df_agent2_win_chance['agent2_Chance']

df_win_chance = pd.merge(df_agent1_win_chance, df_agent2_win_chance, on=['Expansion', 'Exploration', 'Play-out'])
df_win_chance['Total_Games'] = df_win_chance['agent1_Games'] + df_win_chance['agent2_Games']
df_win_chance['Win_Chance'] = (df_win_chance['agent1_Games'] * df_win_chance['agent1_Chance'] + df_win_chance['agent2_Games'] * df_win_chance['agent2_Chance']) / df_win_chance['Total_Games']
df_win_chance = df_win_chance.sort_values(by=['Win_Chance'])

print(df_win_chance)

df_win_chance.to_csv('neural_net_csv/win_chance_data.csv')

df_player_one_bias = df_grouped.groupby(['GameRulesetName', 'Id']).agg({'Games': ['sum'], 'Win_Chance': ['mean']}).reset_index()
df_player_one_bias.columns = df_player_one_bias.columns.get_level_values(0)
df_player_one_bias = df_player_one_bias.sort_values(by=['Win_Chance'])

print(df_player_one_bias)

df_player_one_bias.to_csv('neural_net_csv/player_one_bias_data.csv')

