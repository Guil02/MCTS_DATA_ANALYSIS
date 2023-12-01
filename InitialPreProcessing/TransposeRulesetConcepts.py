import polars as pl


# the goal of this file is to import the data/ludii-database-files/RulesetConcepts.csv file and manipulate
# it such that the first column is every unique RulesetId and all the following columns are the concepts and their value

def run(conceptId=False):
    data = pl.read_csv('new_csv/RulesetConceptsImportant.csv', infer_schema_length=None)
    concepts = pl.read_csv('data/ludii-database-files/Concepts.csv', infer_schema_length=None)
    concepts = concepts.select(['Id', 'Name'])
    data = data.join(concepts, left_on='ConceptId', right_on='Id', how='left')
    data = data.select(['RulesetId', 'ConceptId', 'Value', 'Name'])
    data = data.sort('ConceptId')
    if conceptId:
        data = data.select(['RulesetId', 'ConceptId', 'Value'])
        data = data.pivot(index='RulesetId', columns='ConceptId', values='Value', aggregate_function='first')
    else:
        data = data.select(['RulesetId', 'Name', 'Value'])
        data = data.pivot(index='RulesetId', columns='Name', values='Value', aggregate_function='first')
    data.write_csv('new_csv/RulesetConceptsSorted.csv')


if __name__ == '__main__':
    run()
