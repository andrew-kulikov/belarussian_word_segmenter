import numpy as np
import functools
import pickle
import os.path as op
import io

symbols = [',', '.', '?', '!', '\"', '/', '_', '‚', '¦', '%', '”',
           '{', '}', '[', ']', '(', ')', ':', '#', '@', '^', '\\',
           ';', '»', '«', '“', '„', '№', '+', '*', '$', '…', '™']


class OneGramDict(dict):
    def __init__(self):
        self.gramCount = 0
        with open('unigrams.txt', encoding='utf-8') as f:
            for line in f.readlines():
                (word, count) = line[:-1].split('\t')
                self[word] = int(count)
                self.gramCount += self[word]
        super().__init__()

    def __call__(self, word):
        if word in self:
            return float(self[word]) / self.gramCount
        else:
            return 1.0 / (self.gramCount * 10**(len(word) - 2))


def remove_symbols(s):
    for sym in symbols:
        s = s.replace(sym, ' ')
    return s.lower()


def split_pairs(word):
    return [(word[:i+1], word[i+1:]) for i in range(len(word))]


def form_unigrams():
    with open('train-bel.txt', encoding='utf-8') as f:
        unigrams = {}
        for line in f.readlines():
            line = remove_symbols(line)
            words = line.split()
            for word in words:
                unigrams[word] = unigrams.get(word, 0) + 1
        with open('unigrams.txt', 'w+', encoding='utf-8') as out:
            unigrams = sorted(unigrams.items(), key=lambda x: x[1])
            for t in unigrams:
                s = t[0] + '\t' + str(t[1]) + '\n'
                out.write(s)


def form_bigrams():
    with open('train-bel.txt', encoding='utf-8') as f:
        bigrams = {}
        for line in f.readlines():
            line = remove_symbols(line)
            words = line.split()
            for i in range(len(words) - 1):
                bigram = words[i] + ' ' + words[i + 1]
                bigrams[bigram] = bigrams.get(bigram, 0) + 1
        with open('bigrams.txt', 'w+', encoding='utf-8') as out:
            bigrams = sorted(bigrams.items(), key=lambda x: x[1])
            for t in bigrams:
                s = t[0] + '\t' + str(t[1]) + '\n'
                out.write(s)


def save_obj(obj, name):
    """Saves the model in binary format"""
    with open(op.join('data',  name + '.pkl'), 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Loads the model from binary format"""
    with open(op.join('data', name + '.pkl'), 'rb+') as f:
        return pickle.load(f)


def parse(filename):
    """Read `filename` and parse tab-separated file of word and count pairs."""
    with io.open(filename, encoding='utf-8') as reader:
        lines = (line.split('\t') for line in reader)
        return dict((word, float(number)) for word, number in lines)


def segment(word):
    if not word:
        return []
    all_segmentations = [[first] + segment(rest)
                         for (first, rest) in split_pairs(word)]
    return max(all_segmentations, key=word_seq_fitness)


def word_seq_fitness(words):
    return functools.reduce(lambda x, y: x+y,
                            (np.log10(single_word_prob(w)) for w in words))


single_word_prob = OneGramDict()
