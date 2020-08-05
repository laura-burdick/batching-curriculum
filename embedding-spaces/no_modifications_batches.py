# Laura Burdick (lburdick@umich.edu)
# Default curriculum, basic batching 

from gensim.models import word2vec
import sys
import math
from tqdm import tqdm
from common import *

# Read in command line arguments
if len(sys.argv) < 3:
	print('Takes two arguments: seed of word2vec, number of batches')
	exit(2)
numBatches = int(sys.argv[2])
print('Processing with number of batches:',numBatches)

# Name that these embedding spaces will be saved under
name = '_no_modifications_batches_'+str(numBatches)+'_'

(allSeeds,seed,sentences) = beginning(sys.argv[:-1])
if len(sentences) == 0:
	exit()

# Train a single word2vec embedding space
#
# @param seed
#	The seed word2vec is initialized with
# @param sentences
#	A list of sentences to train on,
#	Each sentence is a list of tokens
#
# @returns model
#	Trained word2vec model
#
def trainWithSeed(seed,sentences):
	model = word2vec.Word2Vec(size=300,window=5,min_count=1,seed=seed,iter=5,workers=1)

	current_index = 0
	for i in tqdm(range(numBatches)):
		left = len(sentences) - current_index
		next_index = current_index+left/(numBatches-i)
		if i!=numBatches:
			next_index = math.ceil(next_index)
		next_index = int(next_index)
		model.build_vocab(sentences[current_index:next_index],update=(True if i>0 else False))
		model.train(sentences[current_index:next_index],total_examples=model.corpus_count,epochs=model.iter)
		current_index = next_index

	return model

# Train all word2vec models
train(trainWithSeed,allSeeds,seed,sentences,name)
