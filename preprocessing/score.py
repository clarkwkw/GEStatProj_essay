import numpy as np
import bisect

def closest_level(levels, scores):
	levels = sorted(levels)
	casted = []
	for s in scores:
		index = bisect.bisect(levels, s) - 1
		if index == -1:
			s = levels[0]
		if index == len(levels) - 1:
			s = levels[len(levels) - 1]
		elif abs(levels[index + 1] - s) <= abs(levels[index] - s):
			s = levels[index + 1]
		else:
			s = levels[index]
		casted.append(s)
	return np.asarray(casted)


# assumes all value in 'scores' are in 'levels'
def histogram(levels, scores):
	h = {s: 0 for s in levels}
	for s in scores:
		h[s] += 1
	return h

def confusion_matrix(levels, truth, predictions):
	truth, predictions = np.reshape(truth, (-1)), np.reshape(predictions, (-1))
	level_to_index = {}

	matrix = np.zeros((len(levels), len(levels)))
	count = 0
	for l in levels:
		level_to_index[l] = count
		count += 1

	for k in range(truth.shape[0]):
		i = level_to_index[truth[k]]
		j = level_to_index[predictions[k]]
		matrix[i, j] += 1

	return matrix

# score_levels: list of possible scores
# truth: list/array of real scores
# predictions: list/array of predicted scores
def calculate_qwk(score_levels, truth, predictions):
	truth = np.reshape(np.asarray(truth), (-1))
	predictions = np.reshape(np.asarray(predictions), (-1))

	if score_levels is not None:
		score_levels = sorted(score_levels)
	else:
		min_score = int(min(min(truth), min(predictions)))
		max_score = int(max(max(truth), max(predictions)))
		print("Inferred from input: min. score = %d, max. score = %d"%(min_score, max_score))
		score_levels = list(range(min_score, max_score + 1))
	
	predictions = closest_level(score_levels, predictions)
	truth = closest_level(score_levels, truth)
	hist_truth = histogram(score_levels, truth)
	hist_predict = histogram(score_levels, predictions)
	confusion = confusion_matrix(score_levels, truth, predictions)

	n_ratings = float(len(truth))

	numerator, denominator = 0.0, 0.0
	for i in range(len(score_levels)):
		for j in range(len(score_levels)):
			level_i, level_j = score_levels[i], score_levels[j]
			d = np.power((level_i - level_j)/(len(score_levels) - 1.0), 2)
			e = hist_truth[level_i]*hist_predict[level_j]/n_ratings
			numerator += d * confusion[i, j]/n_ratings
			denominator += d * e / n_ratings

	return 1.0 - numerator/denominator