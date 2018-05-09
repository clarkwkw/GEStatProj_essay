import os
import numpy as np
from . import utils
import re
import nltk

# Adapted from https://github.com/siyuanzhao/automated-essay-grading/blob/master/data_utils.py

GLOVE_DATA_PATH = "glove/glove.42B.300d.txt"
GLOVE_BIN_PATH = "glove/glove.42B.300d.bin"
GLOVE_VEC_DIM = 300

WORD_IDX, WORD2VEC = None, None

def load_glove():
	global WORD_IDX, WORD2VEC

	if WORD_IDX is None:
		try:
			WORD_IDX, WORD2VEC = utils.pickle_load_large(GLOVE_BIN_PATH)
		except Exception as e:
			print("Fail to load glove from binary, falling back to read the raw txt file")
			
			word2vec = []
			word_idx = {}

			word2vec.append(np.asarray([0]*GLOVE_VEC_DIM))
			count = 1
			with open(GLOVE_DATA_PATH) as f:
				line = f.readline().strip()
				while line:
					word, vec_str = line.split(maxsplit = 1)
					word_idx[word] = count
					vector = np.fromstring(vec_str, sep = " ")
					word2vec.append(vector)
					count += 1
					line = f.readline().strip()

			WORD_IDX, WORD2VEC = word_idx, np.stack(word2vec, axis = 0)

def convert_bin():
	if WORD_IDX is None:
		load_glove()

	utils.pickle_save_large((WORD_IDX, WORD2VEC), GLOVE_BIN_PATH)

def texts_to_idx(texts, maxlen):
	if WORD_IDX is None:
		load_glove()

	E = np.zeros((len(texts), maxlen), dtype = np.int32)
	i = 0
	max_sample_len = 0
	for text in texts:
		wc = 0
		for w in nltk.tokenize.word_tokenize(text):
			if wc >= maxlen:
				break
			
			if w.isalpha():
				E[i, wc] = WORD_IDX.get(w.lower(), 0)
				wc += 1

		i += 1
		max_sample_len = max(max_sample_len, wc)
	print("Max_len:", max_sample_len)
	return E



	