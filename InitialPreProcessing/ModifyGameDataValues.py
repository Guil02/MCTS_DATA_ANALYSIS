import polars as pl


def run():
    df = pl.read_csv('../new_csv/AllGamesWithRulesetConcepts.csv', infer_schema_length=None)

    df = df.with_columns(
        [
            pl.col("agents")
            .str.strip('(')
            .str.strip(')')
            .str.split_exact(" / ", 1)
            .struct.rename_fields(["agent1", "agent2"])
            .alias("fields"),
            pl.col('utilities')
            .str.split_exact(";", 1)
            .struct.rename_fields(["utility_agent1", "utility_agent2"])
            .alias("utilitiesStruct"),
        ]
    ).unnest("fields").unnest('utilitiesStruct').drop(["agents", 'utilities']).to_pandas()
    df.dropna(subset='agent2', inplace=True)

    col = df.pop('agent1')
    df.insert(2, col.name, col)
    col = df.pop('agent2')
    df.insert(3, col.name, col)
    col = df.pop('utility_agent1')
    df.insert(4, col.name, col)
    col = df.pop('utility_agent2')
    df.insert(5, col.name, col)

    df = pl.from_pandas(df)

    df.write_csv('../new_csv/data_with_agents_split.csv')


if __name__ == '__main__':
    run()
