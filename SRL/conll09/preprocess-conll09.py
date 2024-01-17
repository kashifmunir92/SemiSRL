def read_conll(filename):
	data = []
	sentence = []
	with open(filename, 'r') as fp:
		for line in fp:
			if len(line.strip()) > 0:
				sentence.append(line.strip().split())
			else:
				data.append(sentence)
				sentence = []
		if len(sentence) > 0:
			data.append(sentence)
			sentence = []
	return data


def srl2ptb(origin_data):
	srl_data = []
	for sentence in origin_data:
		arg_idx = 0
		for i in range(len(sentence)):
			if sentence[i][12] == 'Y':
				srl_sent = []
				srl_sent.append(['0','<DUMMY>', '<DUMMY>', '<DUMMY>', '<DUMMY>', '_', 
								sentence[i][0], sentence[i][13].split('.')[1],' _', '_'])
				for token in sentence:
					srl_sent.append([token[0], token[1], token[3], token[4], token[5], 
									'_', sentence[i][0], token[14 + arg_idx], '_', '_'])
				srl_data.append(srl_sent)
				arg_idx += 1
	return srl_data


def save(srl_data, path):
	with open(path,'w') as f:
		for sent in srl_data:
			for token in sent:
				f.write('\t'.join(token))
				f.write('\n')
			f.write('\n')


import argparse, os
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--train', default=None)
	argparser.add_argument('--test', default=None)
	argparser.add_argument('--dev', default=None)
	argparser.add_argument('--out_dir', default='processed')
	args, extra_args = argparser.parse_known_args()

	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
	if args.train:
		train_conll = read_conll(args.train)
		train_srl = srl2ptb(train_conll)
		save(train_srl, '%s/train_pro' % args.out_dir)
	if args.dev:
		dev_conll = read_conll(args.dev)
		dev_srl = srl2ptb(dev_conll)
		save(dev_srl, '%s/dev_pro' % args.out_dir)
		os.system('cp %s %s/dev_raw' % (args.dev, args.out_dir))
	if args.test:
		test_conll = read_conll(args.test)
		test_srl = srl2ptb(test_conll)
		save(test_srl, '%s/test_pro' % args.out_dir)
		os.system('cp %s %s/test_raw' % (args.test, args.out_dir))
	
