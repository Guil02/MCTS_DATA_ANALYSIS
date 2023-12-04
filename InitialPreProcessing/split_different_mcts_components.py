import polars as pl


def run():
    df = pl.read_csv('../new_csv/data_with_agents_split.csv', infer_schema_length=None)
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

    df = pl.from_pandas(df)
    print(df)
    df.write_csv('../new_csv/output_dataset.csv')


if __name__ == '__main__':
    run()
