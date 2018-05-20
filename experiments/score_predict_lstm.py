from models import TextRNN
import random
import preprocessing
import numpy as np
import json

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
_max_len = 1000
_max_iter = 300
_batch_size = 128
_exclude_stop_words = True
'''
_qwk_score_levels = np.asarray(list(range(
	preprocessing.kaggle.asap_ranges[_asap_prompt_ids[0]][0],
	preprocessing.kaggle.asap_ranges[_asap_prompt_ids[0]][1] + 1
)))
'''
_qwk_score_levels = np.asarray(list(range(14 * 2 + 1))) / 2

_test_model_dir = "output/lstm_1"

def train():
	print("Loading samples..")
	samples = preprocessing.tp_sample.get_samples(_sample_folder)

	#samples = preprocessing.kaggle.get_samples(_asap_file, _asap_prompt_ids)
	
	print("# samples: %d"%len(samples))

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

		unnormalized_pred, unnormalized_labels = [], []
		for j in range(pred.shape[0]):
			idx = valid_idx[j]
			raw_score = samples[idx].unnormalize(pred[j], _score_components)
			unnormalized_pred.append(raw_score)	

		for idx in valid_idx:
			unnormalized_labels.append(samples[idx].score(_score_components))

		print("\tQWK: %.4f"%preprocessing.score.calculate_qwk(_qwk_score_levels, unnormalized_labels, unnormalized_pred))
		
		model.save("output/lstm_%d"%(i+1))
		with open("output/lstm_%d/vocab.json"%(i+1), "w") as f:
			json.dump(vocab, f, indent = 4)

		exit()

def test():

	print("Loading model..")
	model = TextRNN.load(_test_model_dir)
	with open(_test_model_dir + "/vocab.json", "r") as f:
			vocab = json.load(f)

	print("Loading samples..")
	#samples = preprocessing.tp_sample.get_samples(_sample_folder)

	samples = preprocessing.kaggle.get_samples(_asap_file, _asap_prompt_ids)

	print("# samples: %d"%len(samples))

	sample_texts = [s.text for s in samples]
	sample_labels = np.reshape([s.normalized_score(_score_components) for s in samples], [-1, 1])
	raw_score = np.reshape([s.score(_score_components) for s in samples], [-1, 1])

	sample_vecs = preprocessing.nea.texts_to_vec(sample_texts, vocab, _max_len, exclude_stop_words = _exclude_stop_words)

	print("Predicting..")
	r2_score, pred = model.score(sample_vecs, sample_labels, return_prediction = True)

	print("\tValid R^2: %.4f"%(r2_score))

	unnormalized_pred = []
	for idx in range(pred.shape[0]):
		raw_score = samples[idx].unnormalize(pred[idx], _score_components)
		unnormalized_pred.append(raw_score)	

	print("\tQWK: %.4f"%preprocessing.score.calculate_qwk(_qwk_score_levels, raw_score, unnormalized_pred))

train()
#test()
