import polars as pl
from tqdm import tqdm
import os


def run():
    gameIDs = pl.read_csv('data/ludii-database-files/GameRulesetIds.csv')

    def read_file(path, GameRulesetName, Id, full_df):
        try:
            subdir = os.listdir(path)

            for folder in subdir:
                if folder == '.DS_Store':
                    continue
                data = pl.read_csv(path + '/' + folder + '/raw_results.csv', infer_schema_length=0)
                data = data.with_columns(pl.lit(GameRulesetName).alias('GameRulesetName'))
                data = data.with_columns(pl.lit(Id).alias('Id'))
                data = data.select(['GameRulesetName', 'Id', 'agents', 'utilities'])
                full_df = pl.concat([full_df, data])
        except FileNotFoundError:
            pass

        return full_df

    df = pl.DataFrame()
    for i in tqdm(range(len(gameIDs))):
        df = read_file('data/Out/' + gameIDs['GameRulesetName'][i], gameIDs['GameRulesetName'][i], gameIDs['Id'][i], df)

    df.write_csv('new_csv/AllGames.csv')


if __name__ == '__main__':
    run()
