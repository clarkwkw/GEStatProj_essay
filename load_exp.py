import argparse
import preprocessing
import models
import importlib
from pathlib import PurePosixPath

def parse_args():
	parser = argparse.ArgumentParser(description="Loads a testing script under experiments/")
	parser.add_argument("script", type = str, help = "The name of the script")
	args = parser.parse_args()
	return args

try:
	args = parse_args()
	script_name = PurePosixPath(args.script).stem
	module = importlib.import_module('experiments.'+script_name)
except ImportError:
	print("> Cannot load script '%s', abort."%args.script)
	exit(-1)
