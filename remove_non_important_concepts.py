import polars as pl

important_concepts = pl.read_csv('important_concepts.csv', infer_schema_length=None)
print(important_concepts)
