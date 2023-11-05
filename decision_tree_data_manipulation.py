import polars as pl
from pathlib import Path

df = pl.read_csv('new_csv/output_dataset.csv', infer_schema_length=None)
df = df.filter(df['NumPlayers'] == 2.0)
df = df.select(['GameRulesetName', 'Id', 'agent1_AI_type', 'agent1_Expansion', 'agent2_AI_type', 'agent2_Expansion',
                'utility_agent1', 'utility_agent2'])

if not Path.is_dir(Path('decision_tree_csv')):
    Path.mkdir(Path('decision_tree_csv'))

df.write_csv('decision_tree_csv/expansion_data.csv')
