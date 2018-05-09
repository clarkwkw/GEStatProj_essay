from models import TextRNN
import random
import preprocessing
import numpy as np

_sample_folder = "./samples"
_model_folder = "./output"
_name_filter = ["KK201617T1", "KK201617T2"]

_cnn_ngrams = 3
_learning_rate = 1e-3
_k_fold = 5
_vocab_size = 1000
_embedding_dim = 30
_max_len = 2000
_max_iter = 100

def get_label(sample):
	return sample.think + sample.understand + sample.lang + sample.pres

def main():
	print("Loading samples..")
	samples = preprocessing.tp_sample.get_samples(_sample_folder)
	sample_texts = [s.text for s in samples]
	vocab = preprocessing.nea.create_vocab(sample_texts, vocab_size = _vocab_size)
	sample_vecs = preprocessing.nea.texts_to_vec(sample_texts, vocab, _max_len)
	
	sample_idxs = range(sample_vecs.shape[0])
	batches = preprocessing.batch_data(sample_idxs, _k_fold)

	glove_word_matrix = preprocessing.glove.WORD2VEC

	for i in range(_k_fold):
		print("Fold #%d: training.."%(i + 1))
		valid_idx = batches[i]
		train_idx = []

		for j in range(_k_fold):
			if j != i:
				train_idx.extend(batches[j])

		model = TextRNN(_max_len, _vocab_size, _embedding_dim, _cnn_ngrams, _learning_rate)

		model.train(sample_vecs[train_idx], sample_labels[train_idx], _max_iter, True)

		print("\tValid score: %.4f"%(model.score(sample_vecs[valid_idx], sample_labels[valid_idx])))


main()

