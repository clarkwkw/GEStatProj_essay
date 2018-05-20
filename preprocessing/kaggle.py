import codecs
from . import utils

# Adapted from: https://github.com/nusnlp/nea/blob/master/nea/asap_reader.py
asap_ranges = {
	0: (0, 60),
	1: (2,12),
	2: (1,6),
	3: (0,3),
	4: (0,3),
	5: (0,4),
	6: (0,4),
	7: (0,30),
	8: (0,60)
}

def get_samples(tsv_file_path, prompt_ids):
	samples = []
	with codecs.open(tsv_file_path, mode = 'r', encoding = 'UTF8', errors = 'ignore') as f:
		next(f)
		for line in f:
			tokens = line.strip().split("\t")
			essay_set, essay_id = int(tokens[1]), int(tokens[0])
			if essay_set not in prompt_ids:
				continue
			text = tokens[2]
			score = float(tokens[6])
			samples.append(ASAPSample(essay_set, essay_id, text, score))

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

	def get_identifier(self):
		return str(self.__essay_set) + "-" + str(self.__essay_id)

	def score(self, components = []):
		return self.__score

	def normalized_score(self, components = []):
		return (self.__score - asap_ranges[self.__essay_set][0])/(asap_ranges[self.__essay_set][1] - asap_ranges[self.__essay_set][0])

	def unnormalize(self, score = None, components = []):
		if score is None:
			score = self.__score

		low, high = asap_ranges[self.__essay_set] 

		return score * (high - low) + low