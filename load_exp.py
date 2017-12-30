import argparse
import preprocessing
import models
import importlib

def parse_args():
	parser = argparse.ArgumentParser(description="Loads a testing script under experiements/")
	parser.add_argument("script", type = str, help = "The name of the script")
	args = parser.parse_args()
	return args

try:
	args = parse_args()
	module = importlib.import_module('experiements.'+args.script)
except ImportError:
	print("> Cannot load script '%s', abort."%args.script)
	exit(-1)
