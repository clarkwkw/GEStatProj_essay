from models import TextRNN
import random
import preprocessing
import numpy as np

_sample_folder = "./samples"
_model_folder = "./output"
_name_filter = ["KK201617T1", "KK201617T2"]

_asap_file = "./kaggle_data/training_set_rel3.tsv"

_cnn_ngrams = 3
_learning_rate = 1e-3
_k_fold = 5
_vocab_size = 1000
_embedding_dim = 30
_max_len = 450
_max_iter = 100
_exclude_stop_words = True
def get_label(sample):
	return sample.think + sample.understand + sample.lang + sample.pres

def get_asap_label(sample):
	return sample.score

def main():
	print("Loading samples..")
	#samples = preprocessing.tp_sample.get_samples(_sample_folder)
	#sample_labels = np.reshape([get_label(s) for s in samples], [-1, 1])
	
	samples = preprocessing.kaggle.get_samples(_asap_file)
	sample_labels = np.reshape([get_asap_label(s) for s in samples], [-1, 1])
	
	sample_texts = [s.text for s in samples]
	vocab = preprocessing.nea.create_vocab(sample_texts, exclude_stop_words = _exclude_stop_words, vocab_size = _vocab_size)
	sample_vecs = preprocessing.nea.texts_to_vec(sample_texts, vocab, _max_len, exclude_stop_words = _exclude_stop_words)

	sample_idxs = range(sample_vecs.shape[0])
	batches = preprocessing.batch_data(sample_idxs, _k_fold)

	for i in range(_k_fold):
		print("Fold #%d:"%(i + 1))
		valid_idx = batches[i]
		train_idx = []

		for j in range(_k_fold):
			if j != i:
				train_idx.extend(batches[j])

		print("\tInitializing model..")
		model = TextRNN(_max_len, len(vocab), _embedding_dim, _cnn_ngrams, _learning_rate)

		print("\tTraining..")
		model.train(sample_vecs[train_idx], sample_labels[train_idx], _max_iter, True)

		r2_score, pred = model.score(sample_vecs[valid_idx], sample_labels[valid_idx], return_prediction = True)
		print("\tValid R^2: %.4f"%(r2_score))

		model.save("output/lstm")

		print("\tQWK: %.4f"%preprocessing.score.calculate_qwk(None, sample_labels[valid_idx], pred))

main()

