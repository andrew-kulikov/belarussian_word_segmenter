import os.path as op
import os
import numpy as np
from .processing import load_obj, save_obj, parse, clear_text


class Segmenter(object):
    """Segmenter
    Belarussian word segmentation model
    """
    __ALPHABET = set("ʼ'’йцукенгшўзхфываёпролджэячсмітьбюqwertyuiopasdfghjklzxcvbnm0123456789XVI")
    __UNIGRAMS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'unigrams.txt',
    )
    __BIGRAMS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'bigrams.txt',
    )

    def __init__(self):
        self.unigrams = {}
        self.bigrams = {}
        self.total = 0
        self.max_word_length = 0

    def load(self):
        """Load unigram file
        Try to load model from models/unigrams.pkl, if file does not exist
        load from ./unigrams.txt and create binary file for model
        """
        if not op.exists('data'):
            os.mkdir('data')
        if os.path.isfile(op.join('data', 'unigrams.pkl')):
            self.unigrams.update(load_obj('unigrams'))
        else:
            self.unigrams.update(parse(self.__UNIGRAMS_FILENAME))
            save_obj(self.unigrams, 'unigrams')
        self.total = len(self.unigrams)
        self.max_word_length = max(map(len, self.unigrams.keys()))

    def score(self, word):
        """Probability of the given word"""
        unigrams = self.unigrams
        total = self.total

        if word in unigrams:
            return unigrams[word] / total
        # Penalize words not found in the unigrams according
        # to their length, a crucial heuristic.
        return 1 / (total * 10 ** (len(word) - 1))

    def segment(self, text):
        """Word segmentation
        Implemented Viterbi dynamic pogramming algorithm for uniram model
        Returns a list of words (most possible)
        in the same case and order as original without punctuation
        """
        clean_text = clear_text(text)
        best_edge = [(0, 0)] * (len(clean_text) + 1)
        best_edge[0] = None
        best_score = np.zeros(len(clean_text) + 1)

        # forward step - find the score of the best path to each node
        for word_end in range(1, len(clean_text) + 1):
            # initializes best probability with big number
            best_score[word_end] = 10**10
            start_j = max(0, word_end - self.max_word_length)
            for word_start in range(start_j, word_end):
                word = clean_text[word_start:word_end].lower()
                if word in self.unigrams or len(word) == 1:
                    prob = self.score(word)
                    # computing negative log probability
                    cur_score = best_score[word_start] - np.log10(prob)
                    if cur_score < best_score[word_end]:
                        # saves the best segmentation for a word ending in word_end
                        best_score[word_end] = cur_score
                        best_edge[word_end] = (word_start, word_end)

        words = []
        next_edge = best_edge[-1]
        # backward step - create the best path
        while next_edge:
            word = clean_text[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        return words

    def dest(self):
        """Release memory"""
        del self.unigrams


symbols = ['-', '–', '—', ',', '.', '?', '!', '\"', '/', '_', '‚', '™', '>',
           '{', '}', '[', ']', '(', ')', ':', '#', '@', '^', '%', '\\', '<',
           ';', '»', '«', '“', '„', '№', '+', '*', '$', '…', '”', '~']
