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

			word_idx["<unk>"] = 0
			word2vec.appeend(np.zeros((GLOVE_VEC_DIM)))

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

def texts_to_idx(texts, maxlen = None, return_padded_matrix = True):
	if WORD_IDX is None:
		load_glove()

	E = []
	max_sample_len = 0
	for text in texts:
		wc = 0
		words_idx = []
		for w in nltk.tokenize.word_tokenize(text):
			if maxlen is not None and wc >= maxlen:
				break
			
			if w.isalpha():
				words_idx.append(WORD_IDX.get(w.lower(), 0))
				wc += 1

		E.append(words_idx)
		max_sample_len = max(max_sample_len, wc)
	print("Max_len:", max_sample_len)

	if return_padded_matrix:
		if maxlen is None:
			maxlen = max_sample_len

		for i in range(len(E)):
			if len(E[i]) > maxlen:
				E[i] = E[i][0:maxlen]
			elif len(E[i]) < maxlen:
				E[i].extend([0 for _ in range(maxlen - len(E[i]))])

		E = np.asarray(E, dtype = np.int64)

	return E

# Refer to "A Memory-Augmented Neural Model for Automated Grading" - "Input Representation"
def texts_to_pe(texts, maxlen = None):
	if WORD_IDX is None:
		load_glove()

	E = texts_to_idx(texts, maxlen, return_padded_matrix = False)

	PE = np.zeros((len(texts), GLOVE_VEC_DIM))

	l_initial = np.zeros((GLOVE_VEC_DIM))
	for k in range(GLOVE_VEC_DIM):
		l_initial[k] = 1.0 - (k+1.0)/GLOVE_VEC_DIM

	text_count = 0
	for idx_text in E:
		l_increment = np.zeros((GLOVE_VEC_DIM))
		for k in range(GLOVE_VEC_DIM):
			l_increment[k] = (2.0*(k+1.0)/GLOVE_VEC_DIM - 1)/len(idx_text)

		l = l_initial + l_increment
		for idx_word in idx_text:
			PE[text_count] += l
			l += l_increment

		text_count += 1

	return PE