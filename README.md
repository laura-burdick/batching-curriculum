# To Batch or Not to Batch? Comparing Batching and Curriculum Learning Strategies Across Tasks and Datasets
Laura (Wendlandt) Burdick, Jonathan K. Kummerfeld, Rada Mihalcea

Language and Information Technologies (LIT)

University of Michigan

## Introduction
The code in this repository was used in Chapter 5 of Laura Burdick's Ph.D. thesis. I have tried to document it well, but at the end of the day, it is research code, so if you have any problems using it, please get in touch with Laura Burdick (lburdick@umich.edu).

## Citation Information
If you use this code, please cite the following paper:
```
(citation forthcoming)
```

## Code Included
**embedding-spaces/**: This folder includes code for building the embedding spaces for the different curriculum and batching strategies.
- To train embedding spaces, each script can be run from the command line. Each script requires two command line arguments: the seed of word2vec to use (can either be an int, or the string "all" to train with all ten seeds), and the number of batches to train.
- Dependencies: [https://radimrehurek.com/gensim/][gensim], [https://github.com/tqdm/tqdm][tqdm]
- **common.py**: This is common code required by all the different scripts in the folder, so make sure to look at this first. There are some path variables that need to be set at the top before you can run it.
- **no_modifications_batches.py**: Default curriculum, basic batching
- **no_modifications_batches2.py**: Default curriculum, cumulative batching
- **sorted_ascending_batches.py**: Ascending curriculum, basic batching
- **sorted_ascending_batches2.py**: Ascending curriculum, cumulative batching
- **sorted_descending_batches.py**: Descending curriculum, basic batching
- **sorted_descending_batches2.py**: Descending curriculum, cumulative batching

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation (NSF \#1344257), the Defense Advanced Research Projects Agency (DARPA) AIDA program under grant \#FA8750-18-2-0019, and the Michigan Institute for Data Science (MIDAS). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF, DARPA, or MIDAS.
