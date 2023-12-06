import polars as pl


def run():
    important_concepts = pl.read_csv('../data/important_concepts.csv', infer_schema_length=None)
    concepts = pl.read_csv('../data/ludii-database-files/Concepts.csv', infer_schema_length=None)
    remaining_concepts = important_concepts.join(concepts, left_on='TaxonomyString', right_on='TaxonomyString',
                                                 how='left')
    remaining_concepts = remaining_concepts.select(['Id']).rename({'Id': 'ConceptId'})
    ruleset_concepts = pl.read_csv('../data/ludii-database-files/RulesetConcepts.csv', infer_schema_length=None)
    ruleset_concepts_important = remaining_concepts.join(ruleset_concepts, left_on='ConceptId', right_on='ConceptId',
                                                         how='left')
    ruleset_concepts_important.write_csv('../new_csv/RulesetConceptsImportant.csv')


if __name__ == '__main__':
    run()
