import io
import os.path as op
import os
import numpy as np
import pickle
import time


class Segmenter(object):
    """Segmenter
    Belarussian word segmentation model
    """
    ALPHABET = set("ʼ'’йцукенгшўзхфываёпролджэячсмітьбюqwertyuiopasdfghjklzxcvbnm0123456789XVI")
    UNIGRAMS_FILENAME = op.join(
        op.dirname(op.realpath(__file__)),
        'unigrams.txt',
    )
    BIGRAMS_FILENAME = op.join(
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
        if os.path.isfile('models/unigrams.pkl'):
            self.unigrams.update(self.load_obj('unigrams'))
        else:
            self.unigrams.update(self.parse(self.UNIGRAMS_FILENAME))
            self.save_obj(self.unigrams, 'unigrams')
        self.total = len(self.unigrams)
        self.max_word_length = max(map(len, self.unigrams.keys()))

    def save_obj(self, obj, name):
        """Saves the model in binary format"""
        with open('models/'+ name + '.pkl', 'wb+') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """Loads the model from binary format"""
        with open('models/' + name + '.pkl', 'rb+') as f:
            return pickle.load(f)

    @staticmethod
    def parse(filename):
        "Read `filename` and parse tab-separated file of word and count pairs."
        with io.open(filename, encoding='utf-8') as reader:
            lines = (line.split('\t') for line in reader)
            return dict((word, float(number)) for word, number in lines)

    def score(self, word, previous=None):
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
        in the same case as original without punctuation
        """
        clean_text = self.clean(text)
        best_edge = [(0, 0) for i in range(len(clean_text) + 1)]
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

    @classmethod
    def clean(cls, text):
        """Clean given text
        Return text with non-alphanumeric characters removed. Register ignored
        """
        alphabet = cls.ALPHABET
        letters = (letter for letter in text if letter.lower() in alphabet)
        return ''.join(letters)

    def dest(self):
        """Release memory"""
        del self.unigrams


symbols = ['-', '–', '—', ',', '.', '?', '!', '\"', '/', '_', '‚', '™', '>',
           '{', '}', '[', ']', '(', ')', ':', '#', '@', '^', '%', '\\', '<',
           ';', '»', '«', '“', '„', '№', '+', '*', '$', '…', '”', '~']


def main():
    segmenter = Segmenter()
    segmenter.load()
    i = 0
    start_time = time.time()

    with open('test-bel.txt', encoding='utf-8') as infile, \
            open('kek_gold.txt', 'w+', encoding='utf-8') as outfile:

        for line in infile.readlines():
            line = line.replace(chr(8203), '').replace(' ', '').strip()
            new_line = ' '.join(segmenter.segment(line.strip()))
            res = []
            j = 0
            k = 0
            while True:
                if k < len(line):
                    if line[k] in symbols:
                        if line[k] == '-' or line[k] == '–' or line[k] == '—' or line[k] == '(' or line[k] == '«':
                            res.append(' ')
                        res.append(line[k])
                        k += 1
                        continue
                if k >= len(line) or j >= len(new_line):
                    break
                sym = line[k - 1]
                if (sym == '"' or sym == '«' or sym == '»' or sym == '(' or sym == '%') and new_line[j] == ' ':
                    j += 1
                    continue
                if sym == '»' and line[k] not in symbols:
                    res.append(' ')
                if sym == '—' or sym == '–' and new_line[j] != ' ':
                    res.append(' ')
                res.append(new_line[j])
                if new_line[j] != ' ':
                    k += 1
                j += 1

            outfile.write(''.join(res).strip() + '\n')
            i += 1
            if i == 30:
                break
            if i % 100 == 0:
                print('Progress: ', i, end=' ')
        print('Time elapsed:', time.time() - start_time)
        segmenter.dest()


if __name__ == '__main__':
    main()
