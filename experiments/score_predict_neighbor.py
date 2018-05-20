from models import KNNRegressor
import random
import preprocessing
import numpy as np
import json

_sample_folder = "./samples"
_model_folder = "./output"
_name_filter = ["KK201617T1", "KK201617T2"]
_score_components = ["think", "understand", "lang", "pres"]

_n_neighbors = 5
_k_fold = 5
_qwk_score_levels = np.asarray(list(range(14 * 2 + 1))) / 2

def train():
	print("Loading samples..")
	samples = preprocessing.tp_sample.get_samples(_sample_folder)

	print("# samples: %d"%len(samples))
	sample_labels = np.reshape([s.score(_score_components) for s in samples], [-1, 1])
	sample_texts = [s.text for s in samples]

	print("Preprocessing..")
	sample_pe = preprocessing.glove.texts_to_pe(sample_texts)

	sample_idxs = range(sample_pe.shape[0])
	batches = preprocessing.batch_data(sample_idxs, _k_fold)

	for i in range(_k_fold):
		print("Fold #%d:"%(i + 1))
		valid_idx = batches[i]
		train_idx = []

		for j in range(_k_fold):
			if j != i:
				train_idx.extend(batches[j])

		model = KNNRegressor(n_neighbors = _n_neighbors)

		model.fit(sample_pe[train_idx], sample_labels[train_idx])

		score, pred = model.score(sample_pe[valid_idx], sample_labels[valid_idx], return_prediction = True)

		print("R^2 (valid) = %.4f"%score)

		print("QWK (valid) = %.4f"%preprocessing.score.calculate_qwk(_qwk_score_levels, sample_labels[valid_idx], pred))

train()