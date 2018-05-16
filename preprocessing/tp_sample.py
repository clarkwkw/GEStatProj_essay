import re
import os
import json
from . import utils

score_range = {
	"think": 6,
	"understand": 4,
	"lang": 2,
	"pres": 2 
}

def get_samples(sample_folder):
	try:
		files = os.listdir(sample_folder)
	except FileNotFoundError:
		print("Folder '%s' does not exist, abort."%sample_folder)
		exit(-1)
	samples = []
	for file in files:
		_, ext = os.path.splitext(file)
		name = os.path.basename(file)
		if ext == '.json':
			try:
				samples.append(TPSample(sample_folder + '/' + file))
			except Exception as e:
				raise e
				print("Cannot read sample %s"%name)
	return samples

# adapt from https://stackoverflow.com/questions/9590382/forcing-python-json-module-to-work-with-ascii
def ascii_encode_dict(data):
	ascii_encode = lambda x: x.encode('ascii')
	return dict(map(ascii_encode, pair) for pair in data.items())

class TPSample:
	def __init__(self, path):
		with open(path, "r") as f:
			json_dict = json.load(f, object_hook = ascii_encode_dict)

		self.type = json_dict["type"]
		self.batch_name = json_dict["batch_name"]
		self.batch_no = json_dict["batch_no"]

		self.question = json_dict["question"]

		self.think = json_dict["score"]["think"]
		self.understand = json_dict["score"]["understand"]
		self.lang = json_dict["score"]["lang"]
		self.pres = json_dict["score"]["pres"]
		self.score_dict = json_dict["score"]

		self.text = json_dict["text"]
		self.text = utils.clean_string(self.text)

		if "comment" in json_dict:
			self.comment = json_dict["comment"]
		else:
			self.comment = ""
		
	def get_identifier(self):
		return self.type+'-'+self.batch_name+'-'+self.batch_no

	def normalized_score(self, components):
		total, max_total = 0.0, 0.0
		for c in components:
			total += self.score_dict[c]
			max_total += asap_range[c]

		return 1.0*total/max_total

	def unnormalize(self, score, components):
		max_total = 0.0
		for c in components:
			max_total += asap_range[c]

		return score*max_total