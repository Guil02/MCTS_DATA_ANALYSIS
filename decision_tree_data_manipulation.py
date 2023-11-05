import polars as pl
df = pl.read_csv('new_csv/output_dataset.csv',infer_schema_length = None)
df = df.select(['GameRulesetName','Id','agent1_AI_type','agent1_Expansion','agent2_AI_type','agent2_Expansion','utility_agent1','utility_agent2'])

df.write_csv('expansion_data.csv')