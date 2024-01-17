# -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np
from lib import *


class BaseParser(object):
	def __init__(self, vocab, word_dims, pret_dims, lemma_dims, flag_dims, tag_dims, dropout_dim,
				lstm_layers, lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden, mlp_size, 
				dropout_mlp, unified = True):
		
		pc = dy.ParameterCollection()
		self._vocab = vocab
		self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
		self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs(pret_dims))
		self.flag_embs = pc.lookup_parameters_from_numpy(vocab.get_flag_embs(flag_dims))
		self.lemma_embs = pc.lookup_parameters_from_numpy(vocab.get_lemma_embs(lemma_dims))
		self.tag_embs = pc.lookup_parameters_from_numpy(vocab.get_tag_embs(tag_dims))
		
		self.LSTM_builders = []
		input_dims = word_dims + pret_dims + lemma_dims + flag_dims + tag_dims
		f = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens, pc)
		b = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens, pc)
		self.LSTM_builders.append((f, b))
		for i in xrange(lstm_layers - 1):
			f = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
			b = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
			self.LSTM_builders.append((f, b))
		self.dropout_lstm_input = dropout_lstm_input
		self.dropout_lstm_hidden = dropout_lstm_hidden

		W = orthonormal_initializer(mlp_size, 2 * lstm_hiddens)
		self.mlp_arg_W = pc.parameters_from_numpy(W)
		self.mlp_pred_W = pc.parameters_from_numpy(W)
		self.mlp_arg_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_pred_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_size = mlp_size
		self.dropout_mlp = dropout_mlp

		self.rel_W = pc.add_parameters((vocab.rel_size * (mlp_size +1) , mlp_size + 1), 
										init = dy.ConstInitializer(0.))
		self._unified = unified
		self._pc = pc

		def _emb_mask_generator(seq_len, batch_size):
			ret = []
			for i in xrange(seq_len):
				word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
				tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
				scale = 3. / (2.*word_mask + tag_mask + 1e-12)
				word_mask *= scale
				tag_mask *= scale
				word_mask = dy.inputTensor(word_mask, batched = True)
				tag_mask = dy.inputTensor(tag_mask, batched = True)
				ret.append((word_mask, tag_mask))
			return ret
		self.generate_emb_mask = _emb_mask_generator


	@property 
	def parameter_collection(self):
		return self._pc


	def run(self, word_inputs, lemma_inputs, tag_inputs, pred_golds, rel_targets = None, isTrain = True):
		# inputs, targets: seq_len x batch_size
		def dynet_flatten_numpy(ndarray):
			return np.reshape(ndarray, (-1,), 'F')

		batch_size = word_inputs.shape[1]
		seq_len = word_inputs.shape[0]
		marker = self._vocab.PAD if self._unified else self._vocab.DUMMY
		mask = np.greater(word_inputs, marker).astype(np.float32)
		num_tokens = int(np.sum(mask))

		word_embs = [dy.lookup_batch(self.word_embs, 
									np.where(w < self._vocab.words_in_train, w, self._vocab.UNK)
						) for w in word_inputs]
		pre_embs = [dy.lookup_batch(self.pret_word_embs, w) for w in word_inputs]
		flag_embs = [dy.lookup_batch(self.flag_embs, 
									np.array(w == i + 1, dtype=np.int)
						) for i, w in enumerate(pred_golds)]
		lemma_embs = [dy.lookup_batch(self.lemma_embs, lemma) for lemma in lemma_inputs]
		tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tag_inputs]
		
		if isTrain:
			emb_masks = self.generate_emb_mask(seq_len, batch_size)
			emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
											dy.cmult(lemma, wm), dy.cmult(pos, posm)]) 
							for word, pre, flag, lemma, pos, (wm, posm) in 
								zip(word_embs, pre_embs, flag_embs, lemma_embs, tag_embs, emb_masks)]
			
		else:
			emb_inputs = [dy.concatenate([word, pre, flag, lemma, pos]) 
							for word, pre, flag, lemma, pos in 
								zip(word_embs, pre_embs, flag_embs, lemma_embs, tag_embs)]

		top_recur = dy.concatenate_cols(
						biLSTM(self.LSTM_builders, emb_inputs, batch_size, 
								self.dropout_lstm_input if isTrain else 0., 
								self.dropout_lstm_hidden if isTrain else 0.))
		if isTrain:
			top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

		W_arg, b_arg = dy.parameter(self.mlp_arg_W), dy.parameter(self.mlp_arg_b)
		W_pred, b_pred = dy.parameter(self.mlp_pred_W), dy.parameter(self.mlp_pred_b)
		arg_hidden = leaky_relu(dy.affine_transform([b_arg, W_arg, top_recur]))
		# pred_hidden = leaky_relu(dy.affine_transform([b_pred, W_pred, top_recur]))
		predicates_1D = pred_golds[0]
		pred_recur = dy.pick_batch(top_recur, predicates_1D, dim=1)
		pred_hidden = leaky_relu(dy.affine_transform([b_pred, W_pred, pred_recur]))
		if isTrain:
			arg_hidden = dy.dropout_dim(arg_hidden, 1, self.dropout_mlp)
			# pred_hidden = dy.dropout_dim(pred_hidden, 1, self.dropout_mlp)
			pred_hidden = dy.dropout(pred_hidden, self.dropout_mlp)

		W_rel = dy.parameter(self.rel_W)

		# rel_logits = bilinear(arg_hidden, W_rel, pred_hidden, self.mlp_size, seq_len, batch_size, 
		# 						num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# # (#pred x rel_size x #arg) x batch_size
		
		# flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
		# # (#pred x rel_size) x (#arg x batch_size)

		# predicates_1D = dynet_flatten_numpy(pred_golds)
		# partial_rel_logits = dy.pick_batch(flat_rel_logits, predicates_1D)
		# # (rel_size) x (#arg x batch_size)
		
		rel_logits = bilinear(arg_hidden, W_rel, pred_hidden, self.mlp_size, seq_len, 1, batch_size, 
								num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# (1 x rel_size x #arg) x batch_size
		flat_rel_logits = dy.reshape(rel_logits, (1, self._vocab.rel_size), seq_len * batch_size)
		# (1 x rel_size) x (#arg x batch_size)

		predicates_1D = np.zeros(dynet_flatten_numpy(pred_golds).shape[0])
		partial_rel_logits = dy.pick_batch(flat_rel_logits, predicates_1D)
		# (1 x rel_size) x (#arg x batch_size)

		if isTrain:
			mask_1D = dynet_flatten_numpy(mask)
			mask_1D_tensor = dy.inputTensor(mask_1D, batched = True)
			rel_preds = partial_rel_logits.npvalue().argmax(0)
			targets_1D = dynet_flatten_numpy(rel_targets)
			rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
			rel_accuracy = np.sum(rel_correct)/ num_tokens
			losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
			rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
			return rel_accuracy, rel_loss

		# rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), 
		# 									(self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
		
		rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), 
											(self._vocab.rel_size, 1, seq_len, batch_size), 'F'))
		outputs = []

		# for msk, pred_gold, rel_prob in zip(np.transpose(mask), pred_golds.T, rel_probs):
		# 	msk[0] = 1.
		# 	sent_len = int(np.sum(msk))
		# 	rel_prob = rel_prob[np.arange(len(pred_gold)), pred_gold]
		# 	rel_pred = rel_argmax(rel_prob)
		# 	outputs.append(rel_pred[:sent_len])
		
		for msk, pred_gold, rel_prob in zip(np.transpose(mask), pred_golds.T, rel_probs):
			msk[0] = 1.
			sent_len = int(np.sum(msk))
			rel_prob = rel_prob[np.arange(len(pred_gold)), 0]
			rel_pred = rel_argmax(rel_prob)
			outputs.append(rel_pred[:sent_len])

		return outputs


	def save(self, save_path):
		self._pc.save(save_path)


	def load(self, load_path):
		self._pc.populate(load_path)

