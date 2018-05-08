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
_max_len = 2000
_max_iter = 100

def get_label(sample):
	return sample.think + sample.understand + sample.lang + sample.pres

def main():
	print("Loading glove..")
	preprocessing.glove.load_glove()

	print("Loading samples..")
	samples = preprocessing.tp_sample.get_samples(_sample_folder)
	sample_vects = preprocessing.glove.texts_to_idx([s.text for s in samples], _max_len)
	sample_labels = np.asarray([[get_label(s)] for s in samples])

	sample_idxs = range(sample_vects.shape[0])
	batches = preprocessing.batch_data(sample_idxs, _k_fold)

	glove_word_matrix = preprocessing.glove.WORD2VEC

	for i in range(_k_fold):
		print("Fold #%d: training.."%(i + 1))
		valid_idx = batches[i]
		train_idx = []

		for j in range(_k_fold):
			if j != i:
				train_idx.extend(batches[j])

		model = TextRNN(glove_word_matrix, _max_len, _cnn_ngrams, _learning_rate)

		model.train(sample_vects[train_idx], sample_labels[train_idx], _max_iter, True)

		print("\tValid score: %.4f"%(model.score(sample_vects[valid_idx], sample_labels[valid_idx])))


main()

