# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle
sys.path.append('..')
import dynet as dy
import numpy as np
import models
from lib import Vocab, DataLoader
from test import test
from config import Configurable

import argparse
if __name__ == "__main__":
	np.random.seed(666)
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/default.cfg')
	argparser.add_argument('--model', default='BaseParser')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)

	vocab = Vocab(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
	cPickle.dump(vocab, open(config.save_vocab_path, 'w'))
	parser = Parser(vocab, config.word_dims, config.pret_dims, config.lemma_dims, config.flag_dims, 
					config.tag_dims, config.dropout_emb, config.lstm_layers, config.lstm_hiddens, 
					config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_rel_size, 
					config.dropout_mlp, config.unified)
	data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
	pc = parser.parameter_collection
	trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
	
	global_step = 0
	def update_parameters():
		trainer.learning_rate =config.learning_rate*config.decay**(global_step / config.decay_steps)
		trainer.update()

	epoch = 0
	best_F1 = 0.
	history = lambda x, y : open(os.path.join(config.save_dir, 'valid_history'),'a').write('%.2f %.2f\n'%(x,y))
	while global_step < config.train_iters:
		print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d'%(epoch, )
		epoch += 1
		for words, lemmas, tags, arcs, rels in \
				data_loader.get_batches(batch_size = config.train_batch_size, shuffle = True):
			num = int(words.shape[1]/2)
			words_ = [words[:,:num], words[:,num:]]
			lemmas_ = [lemmas[:,:num], lemmas[:,num:]]
			tags_ = [tags[:,:num], tags[:,num:]]
			arcs_ = [arcs[:,:num], arcs[:,num:]]
			rels_ = [rels[:,:num], rels[:,num:]] 
			for step in xrange(2):
				dy.renew_cg()
				rel_accuracy, loss = parser.run(words_[step], lemmas_[step], tags_[step], arcs_[step], rels_[step])
				loss = loss * 0.5
				loss_value = loss.scalar_value()
				loss.backward()
				sys.stdout.write("Step #%d: Acc: rel %.2f, loss %.3f\r\r" % 
									(global_step, rel_accuracy, loss_value))
				sys.stdout.flush()
			update_parameters()

			global_step += 1
			if global_step % config.validate_every == 0:
				print '\nTest on development set'
				NF1, F1 = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.pro_dev_file, 
								config.raw_dev_file, os.path.join(config.save_dir, 'valid_tmp'), config.prune_num, 
								config.unified, config.disambiguation_accuracy)
				history(NF1, F1)
				if global_step > config.save_after and F1 > best_F1:
					best_F1 = F1
					parser.save(config.save_model_path)

