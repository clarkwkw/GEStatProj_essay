import os
import numpy as np
import preprocessing
import textbook
import nltk
import pandas

"""
Computes the similarity between essays and each chapter of the textbook
The similarity measure is defined as:
1. find out 50 words from each chapter with the most occurence
2. for each essay,
	for a chapter,
		find out the occurence of the 50 words in the essay to construct the word vector,
		normalize both vectors (essay and chapter)
		calculate the dot product
"""

sample_folder = "./samples"
out_file = "similarity.csv"
n_key_vocabs = 30

def normalize(v):
	norm = np.linalg.norm(v)
	if norm != 0:
		return v/norm
	else:
		return v

def dict_to_arr(dict, fields = []):
	result = []
	for field in fields:
		result.append(dict[field])
	result = np.reshape(result, len(result))
	return result

def cal_similarity(v1, v2):
	v1 = normalize(v1)
	v2 = normalize(v2)
	dot_product = np.dot(v1, v2)
	return dot_product

samples = preprocessing.tp_sample.get_samples(sample_folder)
similarity = np.zeros((len(samples), len(textbook._chapter_pg)))
key_vocabs_all = {}
key_vocabs_chapters = []
chapter_vects = []

# Get important vocabs of each chapter
for j in range(len(textbook._chapter_pg)):
	ch = textbook._chapter_pg[j][0]
	key_vocabs = []
	chapter_vect = []
	for (vocab, freq) in textbook.getTopVocabs(ch, n_key_vocabs):
		key_vocabs.append(vocab)
		chapter_vect.append(freq)
		key_vocabs_all[vocab] = 0
	chapter_vect = np.reshape(chapter_vect, len(chapter_vect))
	key_vocabs_chapters.append(key_vocabs)
	chapter_vects.append(chapter_vect)

# For each sample, count the appearance of important vocabs
# Then, calculate cosine similarity
for i in range(len(samples)):
	sample_vocab_freq = dict(key_vocabs_all)
	for token in nltk.word_tokenize(samples[i].text):
		if token in sample_vocab_freq:
			sample_vocab_freq[token] += 1
	for j in range(len(textbook._chapter_pg)):
		sample_vect = dict_to_arr(sample_vocab_freq, key_vocabs_chapters[j])
		similarity[i, j] = cal_similarity(sample_vect, chapter_vects[j])

similarity_df = pandas.DataFrame(similarity, columns = textbook.getChapterTitles())
similarity_df.index = [sample.get_identifier() for sample in samples]

similarity_df.to_csv(out_file)