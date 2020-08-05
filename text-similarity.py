# Laura Burdick (lburdick@umich.edu)
# Code to calculate sentence and phrase similarity for an embedding space

import pickle
import sys
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import spearmanr
from nltk.tokenize import word_tokenize

# SET THESE VARIABLES

# Folder prefix where all the similarity data is stored (can set to empty string if you don't want to use a prefix)
dataFolder = '/local/data/'

# Training data for human activities dataset
# This data can be downloaded from https://lit.eecs.umich.edu/downloads.html, under ``Human Activity Phrase Data".
# We will be using the columns named act1, act2, sim, rel, ma, and pac.
activities_data = dataFolder+"human_activity_phrase_data/gold_pairs.csv"

# Training data for STS Benchmark
# This data can be downloaded from https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark.
# It needs to be formatted into a csv file with columns named sentence1, sentence2, and score.
sts_data =  dataFolder+"stsbenchmark/all_train.csv"

# Training data for SICK
# This data can be downloaded from https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC.
# It needs to be formatted into a csv file with columns named sentence_A, sentence_B, and relatedness_score.
sick_data = dataFolder+"SICK/train.csv"

# Location where the trained embedding spaces are stored
# Filenames should have format WIKI_SENTENCES_{embedding_space_name}_{batch_num}_{word2vec_seed}
embedding_location = dataFolder+'embedding_datasets/baselines/models/'

# Location where you want to store the sentence and phrase similarity results
# Filenames will have format similarity_WIKI_SENTENCES_{embedding_space_name}_{batch_num}.csv
output_location = dataFolder+'embedding_datasets/baselines/lrecExperiments/similarity/'

# Requires one argument: the name of the embedding space that you want to calculate similarity for.
# The embedding space names are the name of the python scripts in the embedding-spaces/ folder.
# For example, the embedding space name for the default curriculum and basic batching is "no_modifications_batches".
# By default, this script runs similarity for all ten seeds of word2vec.
#
if len(sys.argv) < 2:
	print("Takes one argument: name of embedding space")
	exit()
baseline_name = sys.argv[1]
seeds = [2518,2548,2590,29,401,481,485,533,725,777]
print('Processing with seeds:',seeds)

# Load all similarity data
print("Load human activities")
activities = pd.read_csv(activities_data)

print('Load STS')
sts = pd.read_csv(sts_data)

print('Load SICK')
sick = pd.read_csv(sick_data)

# Get average embedding for a set of words in an embedding space model
#
# @param words
#	List of words that you want the average for
# @param model
#	Embedding space model
#
# @returns avg
#	Average embedding formatted as a 300-dimensional list
#
def getAverage(words,model):
	avg = [0]*300
	count = 0
	for word in words:
		if word in model:
			avg += model[word]
			count += 1
	if count > 0:
		avg = [i/count for i in avg]
	
	return avg

# Get cosine similarities (predicted similarities) between two sets of words for an embedding space model
#
# @param words1
#	First list of words
# @param words2
#	Second list of words
# @param model
#	Embedding space model
#
# @returns predicted
#	List of cosine similarities between words1 and words2
#
def getPredicted(words1,words2,model):
	avg1 = getAverage(words1,model)
	avg2 = getAverage(words2,model)

	predicted = cosine_similarity(np.array(avg1).reshape(1,-1),np.array(avg2).reshape(1,-1))[0][0]
	
	return predicted

# For all batch sizes, load embedding space models and calculate similarity results
for batch in [2,3,4,5,10,20,50,100,200,400]:
	print('batch',batch)
	modelFile = embedding_location + 'WIKI_SENTENCES_'+baseline_name+'_'+str(batch)+'_'
	outputFile = output_location + 'similarity_WIKI_SENTENCES_'+baseline_name+'_'+str(batch)
	
	with open(outputFile+'_'+str(batch)+'.csv','w') as output:
		output.write('seed,sim,rel,ma,pac,sts,sick\n')
		for seed in seeds:
			print(seed)

			print('Load pre-trained model')
			model = KeyedVectors.load(modelFile+str(seed))

			print('Calculate activity similarities')
			predicted = []
			for it,row in activities.iterrows():
				words1 = word_tokenize(row['act1'])
				words2 = word_tokenize(row['act2'])
				predicted.append(getPredicted(words1,words2,model))
				
			print("Do Spearman's correlation")
			sim_correlation = spearmanr(predicted,activities['sim']).correlation
			rel_correlation = spearmanr(predicted,activities['rel']).correlation
			ma_correlation = spearmanr(predicted,activities['ma']).correlation
			pac_correlation = spearmanr(predicted,activities['pac']).correlation
			output.write(str(seed)+','+str(sim_correlation)+','+str(rel_correlation)+','+str(ma_correlation)+','+str(pac_correlation)+',')

			print('Calculate sts similarities')
			predicted = []
			for it,row in sts.iterrows():
				words1 = word_tokenize(row['sentence1'])
				words2 = word_tokenize(row['sentence2'])
				predicted.append(getPredicted(words1,words2,model))
				
			print("Do Spearman's correlation")
			correlation = spearmanr(predicted,sts['score']).correlation
			output.write(str(correlation)+',')

			print('Calculate sick similarities')
			predicted = []
			for it,row in sick.iterrows():
				words1 = word_tokenize(row['sentence_A'])
				words2 = word_tokenize(row['sentence_B'])
				predicted.append(getPredicted(words1,words2,model))
				
			print("Do Spearman's correlation")
			correlation = spearmanr(predicted,sick['relatedness_score']).correlation
			output.write(str(correlation)+'\n')
