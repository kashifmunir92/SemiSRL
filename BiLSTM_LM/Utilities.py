import os
import torch
import torch.nn.utils.rnn as rnn_utils


def padd_sentence(sentences):
    sentences.sort(key=lambda x: len(x), reverse=True)
    sentences_length = [len(x) for x in sentences]
    max_length = sentences_length[0]
    for i in range(len(sentences_length)):
        if sentences_length[i] == max_length:
            sentences_length[i] = sentences_length[i] - 1
    sentences = rnn_utils.pad_sequence(sentences,
                                       batch_first=True,
                                       padding_value=0)
    return sentences, sentences_length


def padd_sentence_crf(SequenceTag):
    sentences = []
    tags = []
    for s, t in SequenceTag:
        sentences.append(s)
        tags.append(t)
    sentences.sort(key=lambda x: len(x), reverse=True)
    sentences_length = [len(x) for x in sentences]
    sentences = rnn_utils.pad_sequence(
        sentences, batch_first=True, padding_value=0)
    tags = rnn_utils.pad_sequence(tags, batch_first=True, padding_value=5)
    return sentences, tags, sentences_length


def build_voc_size(sentences, word_2_idx):
    for sentence in sentences:
        for w in sentence.split():
            if w.lower() not in word_2_idx:
                word_2_idx[w.lower()] = len(word_2_idx) + 1


def prepare_sequence(seq, to_ix):
    idxs = [
        torch.tensor(to_ix[w.lower()], dtype=torch.long) for w in seq.split()
    ]
    return torch.tensor(idxs, dtype=torch.long)


def read_file(path):
    path = os.getcwd() + path
    sentences = []
    with open(path, encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences
