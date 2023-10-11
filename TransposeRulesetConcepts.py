import polars as pl


# the goal of this file is to import the data/ludii-database-files/RulesetConcepts.csv file and manipulate
# it such that the first column is every unique RulesetId and all the following columns are the concepts and their value

def run():
    data = pl.read_csv('data/ludii-database-files/RulesetConcepts.csv', infer_schema_length=None)
    data = data.select(['RulesetId', 'ConceptId', 'Value'])
    data = data.sort('ConceptId')
    data = data.pivot(index='RulesetId', columns='ConceptId', values='Value')
    data.write_csv('new_csv/RulesetConceptsSorted.csv')


if __name__ == '__main__':
    run()
