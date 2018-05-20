import os, errno
import re
import sys
import pickle
import unicodedata

# Copied from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
ENGLISH_STOP_WORDS = frozenset([
	"a", "about", "above", "across", "after", "afterwards", "again", "against",
	"all", "almost", "alone", "along", "already", "also", "although", "always",
	"am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
	"any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
	"around", "as", "at", "back", "be", "became", "because", "become",
	"becomes", "becoming", "been", "before", "beforehand", "behind", "being",
	"below", "beside", "besides", "between", "beyond", "bill", "both",
	"bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
	"could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
	"down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
	"elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
	"everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
	"find", "fire", "first", "five", "for", "former", "formerly", "forty",
	"found", "four", "from", "front", "full", "further", "get", "give", "go",
	"had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
	"hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
	"how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
	"interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
	"latterly", "least", "less", "ltd", "made", "many", "may", "me",
	"meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
	"move", "much", "must", "my", "myself", "name", "namely", "neither",
	"never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
	"nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
	"once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
	"ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
	"please", "put", "rather", "re", "same", "see", "seem", "seemed",
	"seeming", "seems", "serious", "several", "she", "should", "show", "side",
	"since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
	"something", "sometime", "sometimes", "somewhere", "still", "such",
	"system", "take", "ten", "than", "that", "the", "their", "them",
	"themselves", "then", "thence", "there", "thereafter", "thereby",
	"therefore", "therein", "thereupon", "these", "they", "thick", "thin",
	"third", "this", "those", "though", "three", "through", "throughout",
	"thru", "thus", "to", "together", "too", "top", "toward", "towards",
	"twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
	"very", "via", "was", "we", "well", "were", "what", "whatever", "when",
	"whence", "whenever", "where", "whereafter", "whereas", "whereby",
	"wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
	"who", "whoever", "whole", "whom", "whose", "why", "will", "with",
	"within", "without", "would", "yet", "you", "your", "yours", "yourself",
	"yourselves"])

def ensure_dir_exist(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def clean_string(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " ( ", string)
	string = re.sub(r"\)", " ) ", string)
	string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	return string.strip().lower()

# source: https://stackoverflow.com/questions/42653386/does-pickle-randomly-fail-with-oserror-on-large-files
def pickle_save_large(obj, filepath):
	"""
	This is a defensive way to write pickle.write, allowing for very large files on all platforms
	"""
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(obj, protocol = 4)
	n_bytes = sys.getsizeof(bytes_out)
	with open(filepath, 'wb') as f_out:
		for idx in range(0, n_bytes, max_bytes):
			f_out.write(bytes_out[idx:idx+max_bytes])


def pickle_load_large(filepath):
	"""
	This is a defensive way to write pickle.load, allowing for very large files on all platforms
	"""
	max_bytes = 2**31 - 1
	input_size = os.path.getsize(filepath)
	bytes_in = bytearray(0)
	with open(filepath, 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	obj = pickle.loads(bytes_in)
	
	return obj

def utf8_to_ascii(s):
	s = unicode(s, "utf-8")
	return unicodedata.normalize('NFKD', s).encode('ascii','ignore')

def _decode_list(data):
	rv = []
	for item in data:
		if isinstance(item, str):
			item = item.encode('ascii', "ignore").decode("ascii")
		elif isinstance(item, list):
			item = _decode_list(item)
		elif isinstance(item, dict):
			item = _decode_dict(item)
		rv.append(item)
	return rv

def _decode_dict(data):
	rv = {}
	for key, value in data.items():
		if isinstance(key, str):
			key = key.encode('ascii', "ignore").decode("ascii")
		if isinstance(value, str):
			value = value.encode('ascii', "ignore").decode("ascii")
		elif isinstance(value, list):
			value = _decode_list(value)
		elif isinstance(value, dict):
			value = _decode_dict(value)
		rv[key] = value
	return rv