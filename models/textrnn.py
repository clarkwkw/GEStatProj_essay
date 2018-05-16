import tensorflow as tf
import numpy as np
from . import utils
import json

class TextRNN:
	def __init__(self, response_length, embedding_n_vocab, embedding_dim, cnn_ngrams, learning_rate, from_save = None):
		self._response_length = response_length
		self._embedding_n_vocab = embedding_n_vocab
		self._embedding_dim = embedding_dim
		self._cnn_ngrams = cnn_ngrams
		self._learning_rate = learning_rate

		self._graph = tf.Graph()
		self._sess = tf.Session(graph = self._graph)
		
		with self._graph.as_default() as g:
			if from_save is None:
				self._query = tf.placeholder(tf.int64, [None, self._response_length], name = "query")
				self._label = tf.placeholder(tf.float32, [None, 1], name = "label")
				
				embedding_mat = tf.get_variable("embedding_matrix", initializer = tf.random_normal([self._embedding_n_vocab, self._embedding_dim]))

				embedded_query = tf.nn.embedding_lookup(embedding_mat, self._query)
				embedded_query = tf.reshape(embedded_query, [-1, 1, self._response_length, self._embedding_dim])
				
				cnn_layer = tf.layers.conv2d(embedded_query, 
					filters = self._embedding_dim, 
					kernel_size = [1, self._cnn_ngrams]
				)

				cnn_layer = tf.reshape(cnn_layer, [-1, self._response_length - cnn_ngrams + 1, self._embedding_dim])

				rnn_input = tf.unstack(cnn_layer, axis = 1)
				lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self._embedding_dim, forget_bias = 1.0, activation = tf.tanh)
				lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 0.5, state_keep_prob = 0.9, output_keep_prob = 0.5)
				lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self._embedding_dim, forget_bias = 1.0, activation = tf.tanh)
				lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 0.5, state_keep_prob = 0.9, output_keep_prob = 0.5)
				
				outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input, dtype = tf.float32)
				output_all = tf.stack(outputs)
				
				mean_over_time_output = tf.reduce_mean(output_all, axis = 0)
				
				self._prediction = tf.layers.dense(mean_over_time_output, 1, activation = tf.sigmoid)
				tf.add_to_collection("pred", self._prediction)

				self._loss = tf.losses.mean_squared_error(self._label, self._prediction)
				tf.add_to_collection("loss", self._loss)

				self._optimizer = tf.train.RMSPropOptimizer(self._learning_rate).minimize(self._loss)
				tf.add_to_collection("optimizer", self._optimizer)

				self._sess.run(tf.global_variables_initializer())

			else:
				saver = tf.train.import_meta_graph(from_save+'/model.ckpt.meta')
				saver.restore(self._sess, from_save+'/model.ckpt')
				self._query = g.get_tensor_by_name("query:0")
				self._label = g.get_tensor_by_name("label:0")
				self._prediction =  tf.get_collection("prediction")[0]
				self._loss =  tf.get_collection("loss")[0]
				self._optimizer =  tf.get_collection("optimizer")[0]

		tf.reset_default_graph()

	def train(self, response_matrix, labels, max_iter, batch_size, valid_response = None, valid_labels = None, print_loss = False):
		with self._graph.as_default() as g:
			dataset = tf.data.Dataset.from_tensor_slices((response_matrix, labels)).batch(batch_size).repeat()
			iter = dataset.make_one_shot_iterator()
			get_next = iter.get_next()

			for i in range(max_iter):
				batch_response, batch_label = self._sess.run(get_next)

				_, loss = self._sess.run(
					[self._optimizer, self._loss], 
					feed_dict = {
						self._query: batch_response,
						self._label: batch_label
					}
				)

				if print_loss and (i + 1)%10 == 0:
					if valid_response is not None and valid_labels is not None:
						loss = self._sess.run(
							self._loss, 
							feed_dict = {
								self._query: valid_response,
								self._label: valid_labels
							}
						)
					print("#%d: %.4f"%(i + 1, loss))

		tf.reset_default_graph()

	def predict(self, response_matrix):
		with self._graph.as_default() as g:
			prediction = self._sess.run(
				self._prediction,
				feed_dict = {
					self._query: response_matrix
				}
			)

		tf.reset_default_graph()

		return prediction

	# R^2 Score
	def score(self, response_matrix, labels, return_prediction = False):
		with self._graph.as_default() as g:
			loss, prediction = self._sess.run(
				[self._loss, self._prediction],
				feed_dict = {
					self._query: response_matrix,
					self._label: labels
				}
			)

		tf.reset_default_graph()

		score = 1 - loss/np.var(labels)
		if not return_prediction:
			return score

		return score, prediction

	def save(self, savedir):
		utils.ensure_dir_exist(savedir)
		with self._graph.as_default() as g:
			saver = tf.train.Saver()
			save_path = saver.save(self._sess, save_path = savedir+'/model.ckpt')
		tf.reset_default_graph()
		
		with open(savedir+"/model_conf.json", "w") as f:
			init_para = {
				"response_length": self._response_length,
				"embedding_n_vocab": self._embedding_n_vocab,
				"embedding_dim": self._embedding_dim,
				"cnn_ngrams": self._cnn_ngrams,
				"learning_rate": self._learning_rate
			}
			f.write(json.dumps(init_para, indent = 4))

	@staticmethod
	def load(savedir):
		init_para = None
		with open(savedir+"/model_conf.json", "r") as f:
			init_para = json.load(f)
		model = TextRNN(from_save = savedir, **init_para)
		return model