# Laura Burdick (lburdick@umich.edu)
# Common code for building all embedding spaces

# SET THESE VARIABLES

# Location of Wikipedia dataset
# Format: one sentence per line, tokenized by space
dataFile = "/local/data/embedding_datasets/curriculum_learning_replication/wiki/wiki_sentences.txt"

# Folder to store resulting embedding spaces
outputFolder = "/local/data/embedding_datasets/baselines/models/"

# Unique dataset identifier
# (Can change this if you want to use a dataset other than Wikipedia)
dataName = 'WIKI_SENTENCES'

# Read command line arguments
# Open datafile
#
# @param arguments
#	List of command line arguments
# 
# @returns (allSeeds,seed,sentences)
#	A tuple where allSeeds is a Boolean indicating whether or not to use all the word2vec seeds,
#	seed is the individual word2vec seed to use (0 if allSeeds is True),
#	and sentences is a list of sentences (where each sentence is a list of tokens)
#
def beginning(arguments):
	if len(arguments) < 2:
		print('Takes one argument: seed of word2vec')
		return (False,0,[])
	allSeeds = False
	if arguments[1] == 'all':
		allSeeds = True
		print('All seeds')
		seed = 0
	else:
		seed = int(arguments[1])
		print('Processing wth seed',seed)

	print('Reading data file')
	with open(dataFile,'r') as data:
		sentences = data.readlines()
		sentences = [i[:-1].split(' ') for i in sentences]

	return (allSeeds,seed,sentences)

# Save embedding space to file
#
# @param name
#	String, file name to use for saving file
# @param seed
#	Seed of word2vec model trained
# @param model
#	word2vec model to save
#
def saveModel(name,seed,model):
	if model == -1:
		return
	model = model.wv #just keep the word vectors (less memory)
	model.save(outputFolder+dataName+name+str(seed))

# Train all word2vec models for a particular curriculum and batching strategy
#
# @param trainWithSeed
#	A function pointer to the particular training function for that curriculum and batching strategy
# @param allSeeds
#	A Boolean indicating whether or not to use all word2vec seeds
# @param seed
#	An integer with a single word2vec seed to use (doesn't matter if allSeeds is True)
# @param sentences
#	A list of sentences to train on (where each sentence is a list of tokens)
# @param
#	String, name to save models under
#
def train(trainWithSeed,allSeeds,seed,sentences,name):
	print('Training word2vec models')
	if allSeeds:
		seeds = [2518, 2548, 2590, 29, 401, 481, 485, 533, 725, 777]
		for seed in seeds:
			print('Seed',seed)
			model = trainWithSeed(seed,sentences)
			saveModel(name,seed,model)
	else:
		model = trainWithSeed(seed,sentences)
		saveModel(name,seed,model)
