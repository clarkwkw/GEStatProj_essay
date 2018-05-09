import tensorflow as tf
import numpy as np

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
				lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self._embedding_dim, forget_bias = 1.0, activation = tf.tanh)
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

	def train(self, response_matrix, labels, max_iter, print_loss = False):
		with self._graph.as_default() as g:
			for i in range(max_iter):
				_, train_loss = self._sess.run(
					[self._optimizer, self._loss], 
					feed_dict = {
						self._query: response_matrix,
						self._label: labels
					}
				)

				if (print_loss + 1)%10 == 0:
					print("#%d: %.4f"%(i + 1, train_loss))

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

	def score(self, response_matrix, labels):
		with self._graph.as_default() as g:
			loss = self._sess.run(
				self._loss,
				feed_dict = {
					self._query: response_matrix,
					self._label: labels
				}
			)

		tf.reset_default_graph()

		return 1 - loss/np.var(labels)
