import pandas as pd
import polars as pl
from pathlib import Path

df = pl.read_csv('../new_csv/data_max.csv', infer_schema_length=None)

agents = ['agent1', 'agent2']
for agent in agents:
    df = df.with_columns(
        [
            pl.col(agent)
            .str.strip('\'')
            .str.split_exact("-", 3)
            .struct.rename_fields(
                [agent + "_AI_type", agent + "_Expansion", agent + "_Exploration", agent + "_Play-out"])
            .alias("fields"),
        ]
        # )
    ).unnest("fields").drop(agent)

df = df.to_pandas()
col = df.pop('agent1_AI_type')
df.insert(2, col.name, col)
col = df.pop('agent1_Expansion')
df.insert(3, col.name, col)
col = df.pop('agent1_Exploration')
df.insert(4, col.name, col)
col = df.pop('agent1_Play-out')
df.insert(5, col.name, col)
col = df.pop('agent2_AI_type')
df.insert(6, col.name, col)
col = df.pop('agent2_Expansion')
df.insert(7, col.name, col)
col = df.pop('agent2_Exploration')
df.insert(8, col.name, col)
col = df.pop('agent2_Play-out')
df.insert(9, col.name, col)

df = df.loc[(df['agent1_AI_type'] == 'MCTS') & (df['agent2_AI_type'] == 'MCTS')]
df = df.drop(columns=['agent1_AI_type', 'agent2_AI_type'])

# df = df.drop(columns=df.columns[df.nunique() == 1])

if not Path.is_dir(Path('../neural_net_csv')):
    Path.mkdir(Path('../neural_net_csv'))

df.to_csv('../neural_net_csv/base_data.csv')

df = pd.read_csv('../neural_net_csv/base_data.csv')

index_columns = df.columns[
    (df.columns != 'Unnamed: 0') & (df.columns != 'Unnamed: 1') & (df.columns != 'utility_agent1') & (
                df.columns != 'utility_agent2')]

print(index_columns)

df_grouped = pd.pivot_table(df, values='utility_agent2', index=index_columns.to_list(),
                            columns=['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

# df_grouped = pd.pivot_table(df, values='utility_agent2', index=['GameRulesetName', 'Id',
#                                                                 'agent1_Expansion', 'agent1_Exploration',
#                                                                 'agent1_Play-out', 'agent2_Expansion',
#                                                                 'agent2_Exploration', 'agent2_Play-out'],
#                             columns=['utility_agent1'], aggfunc={'utility_agent2': 'count'}, fill_value=0)

df_grouped = df_grouped.reset_index()
df_grouped = df_grouped.rename_axis(None, axis=1)

df_grouped = df_grouped.rename(columns={-1.0: 'Losses', 0.0: 'Draws', 1.0: 'Wins'})
df_grouped['Games'] = df_grouped['Losses'] + df_grouped['Draws'] + df_grouped['Wins']
df_grouped['Win_Chance'] = (df_grouped['Wins'] + 0.5 * df_grouped['Draws']) / df_grouped['Games']

print(df_grouped)

print(df_grouped['Games'].value_counts().sort_index(ascending=False))

df_grouped.to_csv('../neural_net_csv/grouped_data.csv')

df_normalized = pd.read_csv('../neural_net_csv/grouped_data.csv')

one_hot_agent1_Expansion = pd.get_dummies(df_normalized['agent1_Expansion'])
one_hot_agent1_Exploration = pd.get_dummies(df_normalized['agent1_Exploration'])
one_hot_agent1_Playout = pd.get_dummies(df_normalized['agent1_Play-out'])

one_hot_agent1 = one_hot_agent1_Expansion.join([one_hot_agent1_Exploration, one_hot_agent1_Playout])
one_hot_agent1 = one_hot_agent1.add_prefix('1_')
one_hot_agent1 = one_hot_agent1.astype(int)

one_hot_agent2_Expansion = pd.get_dummies(df_normalized['agent2_Expansion'])
one_hot_agent2_Exploration = pd.get_dummies(df_normalized['agent2_Exploration'])
one_hot_agent2_Playout = pd.get_dummies(df_normalized['agent2_Play-out'])

one_hot_agent2 = one_hot_agent2_Expansion.join([one_hot_agent2_Exploration, one_hot_agent2_Playout])
one_hot_agent2 = one_hot_agent2.add_prefix('2_')
one_hot_agent2 = one_hot_agent2.astype(int)

df_normalized = df_normalized.drop(columns=['Unnamed: 0', 'agent1_Expansion', 'agent1_Exploration', 'agent1_Play-out',
                                            'agent2_Expansion', 'agent2_Exploration', 'agent2_Play-out'], axis=1)

df_normalized = df_normalized.join([one_hot_agent1, one_hot_agent2])

columns_to_normalize = df_normalized.columns[
    (df_normalized.columns != 'GameRulesetName') & (df_normalized.columns != 'Id') &
    (df_normalized.columns != 'Losses') & (df_normalized.columns != 'Draws') &
    (df_normalized.columns != 'Wins') & (df_normalized.columns != 'Games')]

df_normalized = df_normalized*1

df_normalized[columns_to_normalize] = df_normalized[columns_to_normalize].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

df_normalized = df_normalized.set_index('GameRulesetName')

print(df_normalized)

df_normalized.to_csv('../neural_net_csv/normalized_data.csv')

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