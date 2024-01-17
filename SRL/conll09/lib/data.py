# -*- coding: UTF-8 -*-
from __future__ import division
from collections import Counter
from k_means import KMeans
import numpy as np
import os


class Vocab(object):
	PAD, DUMMY, UNK = 0, 1, 2

	def __init__(self, input_file, pret_file, min_occur_count=2):
		word_counter = Counter()
		lemma_set = set()
		tag_set = set()
		rel_set = set()
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==10), 'Illegal line: %s'%line
					word, lemma, tag = info[1].lower(), info[2].lower(), info[4]
					head, rel = int(info[6]), info[7]
					word_counter[word] += 1
					lemma_set.add(lemma)
					tag_set.add(tag)
					rel_set.add(rel)

		self._id2word = ['<PAD>', '<DUMMY>', '<UNK>']
		self._id2lemma = ['<PAD>', '<DUMMY>', '<UNK>']
		self._id2tag = ['<PAD>', '<DUMMY>', '<UNK>']
		self._id2rel = ['<PAD>']
		for word, count in word_counter.most_common():
			if count > min_occur_count:
				self._id2word.append(word)
		
		self._pret_file = pret_file
		if pret_file or os.path.exists(pret_file):
			self._add_pret_words(pret_file)
		self._id2lemma += list(lemma_set)
		self._id2tag += list(tag_set)
		self._id2rel += list(rel_set)

		reverse = lambda x : dict(zip(x, range(len(x))))
		self._word2id = reverse(self._id2word)
		self._lemma2id = reverse(self._id2lemma)
		self._tag2id = reverse(self._id2tag)
		self._rel2id = reverse(self._id2rel)
		print "Vocab info: #words %d, #lemmas %d #tags %d #rels %d" % \
				(self.vocab_size, self.lemma_size, self.tag_size, self.rel_size)
		
	def _add_pret_words(self, pret_file):
		self._words_in_train_data = len(self._id2word)
		print '#words in training set:', self._words_in_train_data
		words_in_train_data = set(self._id2word)
		with open(pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word = line[0]
					if word not in words_in_train_data:
						self._id2word.append(word)
		print 'Total words:', len(self._id2word)

	def get_pret_embs(self, word_dims=100):
		assert (self._pret_file is not None), "No pretrained file provided."
		if self._pret_file is None or not os.path.exists(self._pret_file):
			print "Pretrained embedding randomly initialized."
			return np.random.randn(len(self._id2word), word_dims).astype(np.float32)
		embs = [[]] * len(self._id2word) 
		with open(self._pret_file) as f:
			for line in f.readlines():
				line = line.strip().split()
				if line:
					word, data = line[0], line[1:]
					embs[self._word2id[word]] = data
		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		pret_embs = np.array(embs, dtype=np.float32)
		return pret_embs / np.std(pret_embs)

	def get_flag_embs(self, flags_dims):
		embs = np.random.randn(2, flags_dims).astype(np.float32)
		return embs / np.std(embs)

	def get_lemma_embs(self, lemma_dims):
		embs = np.random.randn(self.lemma_size, lemma_dims).astype(np.float32)
		return embs / np.std(embs)

	def get_word_embs(self, word_dims):
		embs = np.random.randn(self.words_in_train, word_dims).astype(np.float32)
		return embs / np.std(embs)

	def get_tag_embs(self, tag_dims):
		embs = np.random.randn(self.tag_size, tag_dims).astype(np.float32)
		return embs / np.std(embs)

	def word2id(self, xs):
		if isinstance(xs, list):
			return [self._word2id.get(x, self.UNK) for x in xs]
		return self._word2id.get(xs, self.UNK)
	
	def id2word(self, xs):
		if isinstance(xs, list):
			return [self._id2word[x] for x in xs]
		return self._id2word[xs]

	def lemma2id(self, xs):
		if isinstance(xs, list):
			return [self._lemma2id.get(x, self.UNK) for x in xs]
		return self._lemma2id.get(xs, self.UNK)
	
	def id2lemma(self, xs):
		if isinstance(xs, list):
			return [self._id2lemma[x] for x in xs]
		return self._id2lemma[xs]

	def rel2id(self, xs):
		if isinstance(xs, list):
			return [self._rel2id[x] for x in xs]
		return self._rel2id[xs]

	def id2rel(self,xs):
		if isinstance(xs, list):
			return [self._id2rel[x] for x in xs]
		return self._id2rel[xs]

	def tag2id(self, xs):
		if isinstance(xs, list):
			return [self._tag2id.get(x, self.UNK) for x in xs]
		return self._tag2id.get(xs, self.UNK)

	@property 
	def words_in_train(self):
		return self._words_in_train_data

	@property
	def vocab_size(self):
		return len(self._id2word)

	@property
	def lemma_size(self):
		return len(self._id2lemma)

	@property
	def tag_size(self):
		return len(self._id2tag)

	@property
	def rel_size(self):
		return len(self._id2rel)

class DataLoader(object):
	def __init__(self, input_file, n_bkts, vocab):
		sents = []
		sent = []
		with open(input_file) as f:
			for line in f.readlines():
				info = line.strip().split()
				if info:
					assert(len(info)==10), 'Illegal line: %s'%line
					word = vocab.word2id(info[1].lower())
					lemma = vocab.lemma2id(info[2].lower())
					tag = vocab.tag2id(info[4])
					head, rel = int(info[6]), vocab.rel2id(info[7])
					sent.append([word, lemma, tag, head, rel])
				else:
					sents.append(sent)
					sent = []
		
		len_counter = Counter()
		for sent in sents:
			len_counter[len(sent)] += 1	
		self._bucket_sizes = KMeans(n_bkts, len_counter).splits
		self._buckets = [[] for i in xrange(n_bkts)]
		len2bkt = {}
		prev_size = -1
		for bkt_idx, size in enumerate(self._bucket_sizes):
			len2bkt.update(zip(range(prev_size+1, size+1), [bkt_idx] * (size - prev_size)))
			prev_size = size

		self._record = []
		for sent in sents:
			bkt_idx = len2bkt[len(sent)]
			self._buckets[bkt_idx].append(sent)
			idx = len(self._buckets[bkt_idx]) - 1
			self._record.append((bkt_idx, idx))

		for bkt_idx, (bucket, size) in enumerate(zip(self._buckets, self._bucket_sizes)):
			self._buckets[bkt_idx] = np.zeros((size, len(bucket), 5), dtype=np.int32)
			for idx, sent in enumerate(bucket):
				self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)
	@property
	def idx_sequence(self):
		return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]
	
	def get_batches(self, batch_size, shuffle= True):
		batches = []
		for bkt_idx, bucket in enumerate(self._buckets):
			bucket_len = bucket.shape[1]
			n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
			n_splits = max(n_tokens // batch_size, 1)
			range_func = np.random.permutation if shuffle else np.arange
			for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
				batches.append((bkt_idx, bkt_batch))
		
		if shuffle:
			np.random.shuffle(batches)

		for bkt_idx, bkt_batch in batches:
			word_inputs = self._buckets[bkt_idx][:,bkt_batch, 0]
			lemma_inputs = self._buckets[bkt_idx][:,bkt_batch, 1]
			tag_inputs = self._buckets[bkt_idx][:,bkt_batch, 2]
			arc_targets = self._buckets[bkt_idx][:,bkt_batch, 3]
			rel_targets = self._buckets[bkt_idx][:,bkt_batch, 4]
			yield word_inputs, lemma_inputs, tag_inputs, arc_targets, rel_targets

