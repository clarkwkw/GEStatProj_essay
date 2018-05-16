from models import TextRNN
import random
import preprocessing
import numpy as np

_sample_folder = "./samples"
_model_folder = "./output"
_name_filter = ["KK201617T1", "KK201617T2"]
_score_components = ["think", "understand", "lang", "pres"]

_asap_file = "./kaggle_data/training_set_rel3.tsv"
_asap_prompt_ids = [3]

_cnn_ngrams = 3
_learning_rate = 1e-3
_k_fold = 5
_vocab_size = 4000
_embedding_dim = 30
_max_len = 400
_max_iter = 300
_batch_size = 128
_exclude_stop_words = False

def normalize(labels):
	return labels/14.0

def get_asap_label(sample):
	return sample.score

def main():
	print("Loading samples..")
	#samples = preprocessing.tp_sample.get_samples(_sample_folder)

	samples = preprocessing.kaggle.get_samples(_asap_file, _asap_prompt_ids)
	
	sample_labels = np.reshape([s.normalized_score(_score_components) for s in samples], [-1, 1])
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
		model.train(sample_vecs[train_idx], sample_labels[train_idx], _max_iter, _batch_size, valid_response = sample_vecs[valid_idx], valid_labels = sample_labels[valid_idx], print_loss = True)

		r2_score, pred = model.score(sample_vecs[valid_idx], sample_labels[valid_idx], return_prediction = True)

		print("\tValid R^2: %.4f"%(r2_score))

		unnormalized_pred = []
		for i in range(pred.shape[0]):
			idx = valid_idx[i]
			raw_score = samples[idx].unnormalize(pred[i], _score_components)
			unnormalized_pred.append(raw_score)	

		model.save("output/lstm_%d"%(i+1))

		print("\tQWK: %.4f"%preprocessing.score.calculate_qwk(None, sample_labels[valid_idx], unnormalized_pred))

main()

