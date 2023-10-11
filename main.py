import ReadGameData
import joinFiles
import ModifyGameDataValues
import TransposeRulesetConcepts
import remove_non_important_concepts
from pathlib import Path

print('Starting')
if not Path.is_file(Path('new_csv/AllGames.csv')):
    ReadGameData.run()
if not Path.is_file(Path('new_csv/RulesetConceptsImportant.csv')):
    remove_non_important_concepts.run()
if not Path.is_file(Path('new_csv/RulesetConceptsSorted.csv')):
    TransposeRulesetConcepts.run()
if not Path.is_file(Path('new_csv/AllGamesWithRulesetConcepts.csv')):
    joinFiles.run()
if not Path.is_file(Path('new_csv/data_with_agents_split.csv')):
    ModifyGameDataValues.run()
print('Done')
