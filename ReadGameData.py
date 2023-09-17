import polars as pl
from tqdm import tqdm

gameIDs = pl.read_csv('data/ludii-database-files/GameRulesetIds.csv')

# we are gonna read in game data for all the games in gameIDs.
# The data is stored in data/Out/<GameRulesetName> and in this folder there are multiple other folders
# which each contain one raw_results.csv file. I want to read in all these files into 1 polars dataframe where
# There are 4 columns: GameRulesetName, Id, agents, utilities. The first 2 columns are from the gameIds dataframe
# and the last 2 columns are from the raw_results.csv file.

# The names of the folders in data/Out are equal to the first column in gameIDs
# So we can loop over the first column of gameIDs and read in the raw_results.csv file from the corresponding folder

import os


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

print(df)
df.write_csv('AllGames.csv')
