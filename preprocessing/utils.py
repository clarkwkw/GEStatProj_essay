import os, errno
import re
import sys
import pickle
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