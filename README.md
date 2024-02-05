# Monte Carlo tree search data analysis

This project aims to implement models that based on certain concepts of a game in the LUDII framework to determine what
the best variation of MCTS is for that game.

## Current State

The project currently consists of reading in provided data sets and manipulating them such as to have a useful data
set. The resulting data sets themselves are too large to push to github so they are only accessible by run the code.
This luckily does not take too long. To retrieve the data set you only need to run the [main.py](main.py) file. This
will generate the data set into the [new_csv](new_csv) folder. This data was further manipulated and can be found in
the [training_data](training_data) folder.

The models can be found in the [models](models) folder. All the models except for the multi-armed bandit can be run by
unpickling them. The multi armed bandit requires the user to import the csv files containing the weights. These can them
be passed to the model which is a python class. To predict values with the model, the data provided to the model must be
in the same format as [components regresssion.csv](training_data/components_regression.csv). Then model will then
automatically format the data to make correct predictions.

## Authors

* **Bams Guillaume** - [Guillaume](https://github.com/Guil02) - Developer
* **Persoon Max** - [Max](https://github.com/MaxPersoon) - Developer
* **Rietjens Marco** - [Marco](https://github.com/Rytjens) - Developer
* **Sladić, Dimitar** - [Dimitar](https://github.com/Sladic) - Developer
* **Stefanov Stefan** - [Stefan](https://github.com/StefanStefanov741) - Developer

## References

* Piette, E., Soemers, D. J., Stephenson, M., Sironi, C. F., Winands, M. H., & Browne, C. (2019). Ludii--The Ludemic
  General Game System. arXiv preprint arXiv:1905.05013.