import preprocessing
import random
from models import SVR
import numpy as np
import textbook

"""
Train a model to classify news and essays into different sections with one model per section
For each model, the input would be word vectors, and a number ranging 0 - 1 would be the output, 
indicating the likelihood of the article belonging to the section.

1. Find out the top 1000 occuring words in textbook
2. Reserve 20% of the articles from Guardian as validation set, the others as training set
3. For each section, train a model by, 
	feeding the vectors into the model with labels 0/1, 
	indicating whether the article belongs to the section.
4. After the model of a section is trained, check its accuracy by,
	feeding the vectors of validation set into the model,
	if the prediction is greater than / equal to 0.5,
	consider the article to be in the section.
	Compare the prediction and the true label.
"""

_essays_dir = "./samples"
_n_key_vocabs = 1000
_ngram_rng = (1, 1)

_save_dir = "./output"

_news_dir = "./news_crawler/guardian/texts"
_min_word_count = 400
_train_ratio = 0.8
_max_thread = 4
_textbook_words = 1000
_reduction = None
_reduce_n_attr = 1000
_max_sample_count = None
_stem_words = True

# Neural Network Parameters
_learning_rate = 0.001
_max_iter = 100000
_valid_step = 100
_hidden_nodes = [20]

_section_filter = ["business", "education", "science", "technology", "higher-education-network", "environment", "global-development", "culture", "politics"]
_section_group_map = {
	"education": "education",
	"science": "science & technology",
	"technology": "science & technology",
	"higher-education-network": "education",
	"environment": "environment",
	"global-development": "global-development",
	"business": "business",
	"culture": "culture",
	"politics": "politics"
}


def get_section(sample):
	return sample.section


print("Reading samples.. ")
news_samples = preprocessing.news_sample.get_samples_multithread(_news_dir, _max_thread, _max_sample_count)

print("Preprocessing.. ")
news_samples = [sample for sample in news_samples if sample.word_count >= _min_word_count and sample.section in _section_filter]

random.shuffle(news_samples)
n_samples = len(news_samples)
_sections = _section_filter
if _section_group_map is not None:
	_sections = {section: True for section in _section_group_map.values()}
	_sections = list(_sections.keys())
	print("Grouped sections:", _sections)
	for sample in news_samples:
		sample.section = _section_group_map[sample.section]
	
train_samples = news_samples[0:int(n_samples*_train_ratio)]
test_samples = news_samples[int(n_samples*_train_ratio):n_samples]

print("Samples distribution:", preprocessing.samples_statistics(news_samples, _sections, get_section))
print("Train set distribution:", preprocessing.samples_statistics(train_samples, _sections, get_section))
print("Test set distribution:", preprocessing.samples_statistics(test_samples, _sections, get_section))

train_texts = [sample.text for sample in train_samples]
test_texts = [sample.text for sample in test_samples]

train_matrix, test_matrix, words = preprocessing.preprocess(train_texts, test_texts, selection = "tfidf", select_top = _textbook_words, savedir = _save_dir, words_src = "textbook", normalize_flag = False, reduction = _reduction, reduce_n_attr = _reduce_n_attr, stem_words = _stem_words)

for section in _sections:
	train_labels = preprocessing.samples_to_binary(train_samples, [section], get_section)
	test_labels = preprocessing.samples_to_binary(test_samples, [section], get_section)

	model = SVR()
	print("Training for %s section.. "%section)

	model.train(train_matrix, train_labels)
	predict = model.predict(test_matrix)
	accuracy = 0
	for i in range(predict.shape[0]):
		if predict[i] >= 0.5:
			predict[i] = 1
		else:
			predict[i] = 0
		if predict[i] == test_labels[i]:
			accuracy += 1.0
	accuracy /= predict.shape[0]
	model.save("%s/%s/"%(_save_dir,section))
	print("Accuracy: %.3f"%accuracy)

