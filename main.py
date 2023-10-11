import ReadGameData
import joinFiles
import ModifyGameDataValues
import TransposeRulesetConcepts
import remove_non_important_concepts
from pathlib import Path

print('Starting')
run_consecutive = False
if not Path.is_file(Path('new_csv/AllGames.csv')):
    run_consecutive = True
    ReadGameData.run()
if not Path.is_file(Path('new_csv/RulesetConceptsImportant.csv')) or run_consecutive:
    run_consecutive = True
    remove_non_important_concepts.run()
if not Path.is_file(Path('new_csv/RulesetConceptsSorted.csv')) or run_consecutive:
    run_consecutive = True
    TransposeRulesetConcepts.run(False)
if not Path.is_file(Path('new_csv/AllGamesWithRulesetConcepts.csv')) or run_consecutive:
    run_consecutive = True
    joinFiles.run()
if not Path.is_file(Path('new_csv/data_with_agents_split.csv')) or run_consecutive:
    run_consecutive = True
    ModifyGameDataValues.run()
print('Done')
