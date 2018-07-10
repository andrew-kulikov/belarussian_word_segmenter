from lib import form_bigrams, form_unigrams, Segmenter, SYMBOLS
import time


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
                    if line[k] in SYMBOLS:
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
                if sym == '»' and line[k] not in SYMBOLS:
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
