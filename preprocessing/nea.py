import nltk
import re
import numpy as np

# adapt from https://github.com/nusnlp/nea/blob/master/nea/asap_reader.py
NUM_REGEX = re.compile(r'^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
	return bool(NUM_REGEX.match(token))

def create_vocab(texts, exclude_stop_words = True, vocab_size = None):
	vocab_freq = {}

	for text in texts:
		for token in nltk.tokenize.word_tokenize(text):
			token = token.lower()
			if token.isalpha():
				vocab_freq[token] = vocab_freq.get(token, 0) + 1

	print("# Unique words: %d"%len(vocab_freq))

	sorted_word_freq = sorted(vocab_freq.items(), key = lambda t: t[1], reverse = True)

	st_idx, end_idx = 0, min(vocab_size if vocab_size is not None else float("inf"), len(vocab_freq))

	vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
	vocab_next_idx = len(vocab)
	for w, _ in sorted_word_freq[st_idx:end_idx]:
		vocab[w] = vocab_next_idx
		vocab_next_idx += 1

	return vocab

def texts_to_vec(texts, vocab, seq_len):
	vecs = []
	n_total_words, n_unk, n_num = 0, 0, 0
	vecs = np.zeros((len(texts), seq_len), dtype = np.int32)

	text_count = 0
	for text in texts:
		token_count = 0
		for token in nltk.tokenize.word_tokenize(text):
			if token_count >= seq_len:
				break

			token = token.lower()
			if is_number(token):
				vecs[text_count, token_count] = vocab['<num>']
				n_num += 1
				token_count += 1

			elif token.isalpha():
				if token in vocab:
					vecs[text_count, token_count] = vocab[token]
				else:
					vecs[text_count, token_count] = vocab['<unk>']
					n_unk += 1
				token_count += 1

		text_count += 1
		n_total_words += token_count

	print("# <num>: %d"%n_num)
	print("# <unk>: %d"%n_unk)
	print("# total words: %d"%n_total_words)

	return vecs



