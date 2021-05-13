# To Batch or Not to Batch? Comparing Batching and Curriculum Learning Strategies Across Tasks and Datasets
Laura (Wendlandt) Burdick, Jonathan K. Kummerfeld, Rada Mihalcea

Language and Information Technologies (LIT)

University of Michigan

## Introduction
The code in this repository was used in Chapter 5 of Laura Burdick's Ph.D. thesis. I have tried to document it well, but at the end of the day, it is research code, so if you have any problems using it, please get in touch with Laura Burdick (lburdick@umich.edu).

## Citation Information
If you use this code, please cite the following paper:
```
@phdthesis{burdickThesis2020,
  author  = "Burdick, Laura",
  title   = "Understanding Word Embedding Stability Across Languages and Applications",
  school  = "University of Michigan",
  year    = "2020"
}
```

## Code Included
**embedding-spaces/**: This folder includes code for building the embedding spaces for the different curriculum and batching strategies.
- To train embedding spaces, each script can be run from the command line. Each script requires two command line arguments: the seed of word2vec to use (can either be an int, or the string "all" to train with all ten seeds), and the number of batches to train.
- Dependencies: [gensim](https://radimrehurek.com/gensim/), [tqdm](https://github.com/tqdm/tqdm)
- **common.py**: This is common code required by all the different scripts in the folder, so make sure to look at this first. There are some path variables that need to be set at the top before you can run it.
- **no_modifications_batches.py**: Default curriculum, basic batching
- **no_modifications_batches2.py**: Default curriculum, cumulative batching
- **sorted_ascending_batches.py**: Ascending curriculum, basic batching
- **sorted_ascending_batches2.py**: Ascending curriculum, cumulative batching
- **sorted_descending_batches.py**: Descending curriculum, basic batching
- **sorted_descending_batches2.py**: Descending curriculum, cumulative batching

**text-similarity.py**: Code to calculate sentence and phrase similarity for an embedding space
- This script can be run from the command line. It takes one argument: the name of the embedding space that you want to calculate similarity for. The embedding space names are the names of the python scripts in the embedding-spaces/ folder. For example, the embedding space name for the default curriculum and basic batching is "no_modifications_batches". By default, this script runs similarity for all ten seeds of word2vec.
- Dependencies: [gensim](https://radimrehurek.com/gensim/), [sklearn](https://scikit-learn.org/), [scipy](https://www.scipy.org/), [nltk](https://www.nltk.org/)
- Before running this script, there are some variables at the top of the script that you need to set. You will need to make sure that you have all the similarity datasets downloaded and formatted correctly (more information in the python script).

**pos-tagging/**: Code to calculate part-of-speech tagging results for an embedding space
- **pos-tagging.sh**: This script can be run from the command line. It takes three arguments: the name of the embedding space that you want to calculate part-of-speech tagging for, the seed of word2vec that you want to use, and the dataset that you want to use for part-of-speech tagging. The embedding space names are the names of the python scripts in the embedding-spaces/ folder. For example, the embedding space name for the default curriculum and basic batching is "no_modifications_batches". The word2vec seed can be either "all" (to use all the word2vec seeds) or a specific integer. The dataset name should be either "all" (to run both datasets), "answers", or "email".
- Dependencies: [numpy](https://numpy.org/), [dynet](https://github.com/clab/dynet)
- Before running this script, there are some variables at the top of the script that you need to set. You will need to make sure that you have the part-of-speech tagging datasets downloaded (more information in the script).
- **src/**: Base code for part-of-speech tagger, written by [Jonathan Kummerfeld](http://jkk.name/) (jkummerf@umich.edu). This is a slight variation of code released in a tutorial by Jonathan, available [here](https://jkk.name/neural-tagger-tutorial/).

To do text classification, download [fastText](https://fasttext.cc/).

To calculate stability, you can find documented code [here](https://github.com/laura-burdick/embeddingStability).

## Acknowledgements
This material is based in part upon work supported by the National Science Foundation (NSF \#1344257), the Defense Advanced Research Projects Agency (DARPA) AIDA program under grant \#FA8750-18-2-0019, and the Michigan Institute for Data Science (MIDAS). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF, DARPA, or MIDAS.
