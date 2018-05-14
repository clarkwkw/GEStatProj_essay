import codecs
from . import utils

# Adapted from: https://github.com/nusnlp/nea/blob/master/nea/asap_reader.py
def get_samples(tsv_file_path):
	samples = []
	with codecs.open(tsv_file_path, mode = 'r', encoding = 'UTF8', errors = 'ignore') as f:
		next(f)
		for line in f:
			tokens = line.strip().split("\t")
			essay_set, essay_id = tokens[1], tokens[0]
			text = tokens[2]
			score = float(tokens[6])
			samples.append(ASAPSample(essay_set, essay_id, text, score))

	print("# samples: %d"%len(samples))
	return samples


class ASAPSample:
	def __init__(self, essay_set, essay_id, text, score):
		self.__text = utils.clean_string(text)
		self.__score = score
		self.__essay_id = essay_id
		self.__essay_set = essay_set
		
	@property 
	def text(self):
		return self.__text

	@property 
	def score(self):
		return self.__score

	def get_identifier(self):
		return self.__essay_set + "-" + self.__essay_id