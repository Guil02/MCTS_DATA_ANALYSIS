import polars as pl


def run():
    AllGames = pl.read_csv('../new_csv/AllGames.csv', infer_schema_length=None)
    RulesetConceptsSorted = pl.read_csv('../new_csv/RulesetConceptsSorted.csv', infer_schema_length=None)

    final_df = AllGames.join(RulesetConceptsSorted, left_on='Id', right_on='RulesetId', how='left')
    final_df.write_csv('../new_csv/AllGamesWithRulesetConcepts.csv')


if __name__ == '__main__':
    run()
