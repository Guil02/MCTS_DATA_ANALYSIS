from data_formatting import ReadGameData, remove_non_important_concepts, ModifyGameDataValues, joinFiles, \
    TransposeRulesetConcepts
from pathlib import Path

print('Starting')
if not Path.is_dir(Path('new_csv')):
    Path.mkdir(Path('new_csv'))
run_consecutive = False
print("Starting game collection into 1 file")
if not Path.is_file(Path('new_csv/AllGames.csv')):
    run_consecutive = True
    ReadGameData.run()
print("Starting to remove non important concepts")
if not Path.is_file(Path('new_csv/RulesetConceptsImportant.csv')) or run_consecutive:
    run_consecutive = True
    remove_non_important_concepts.run()
print("Starting to transpose ruleset concepts")
if not Path.is_file(Path('new_csv/RulesetConceptsSorted.csv')) or run_consecutive:
    run_consecutive = True
    TransposeRulesetConcepts.run(False)
print("Starting to join files")
if not Path.is_file(Path('new_csv/AllGamesWithRulesetConcepts.csv')) or run_consecutive:
    run_consecutive = True
    joinFiles.run()
print("Starting to split agents and utilities into separate columns")
if not Path.is_file(Path('new_csv/data_with_agents_split.csv')) or run_consecutive:
    run_consecutive = True
    ModifyGameDataValues.run()
print('Done')
