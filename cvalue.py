#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
- Frantzi, Ananiadou, Mima (2000)- Automatic Recognition of Multi-Word Terms -
  the C-value-NC-value Method
- Barrón-Cedeño, Sierra, Drouin, Ananiadou (2009)- An Improved Term Recognition
  Method for Spanish
"""

from __future__ import division
from collections import defaultdict
from math import log
import nltk
import evaluation


def load_domain():
    with open('corpora/small_domain.txt', 'r') as f:
        corp = f.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in corp.split('./Fp')]
    tagged_sents = [s.split() for s in tagged_sents]
    tagged_sents = [[tuple(w.split('/')) for w in s] for s in tagged_sents]
    return tagged_sents


def chunk_sents(tagged_sents, pos_pattern):
    chunk_freq_dict = defaultdict(int)
    chunker = nltk.RegexpParser(pos_pattern)
    for sent in tagged_sents:
        for chk in chunker.parse(sent).subtrees():
            if str(chk).startswith('(TC'):
                phrase = chk.__unicode__()[4:-1]
                chunk_freq_dict[phrase] += 1
    return chunk_freq_dict


def min_freq_filter(chunk_freq_dict, min_freq):
    chunk_freq_dict = \
        dict([p for p in chunk_freq_dict.items() if p[1] >= min_freq])
    return chunk_freq_dict


def remove_str_postags(tagged_str):
    stripped_str = ' '.join([w.rsplit('/', 1)[0] for w in tagged_str.split()])
    return stripped_str


def remove_dict_postags(chunk_freq_dict):
    new_dict = {}
    for phrase in chunk_freq_dict.keys():
        new_str = remove_str_postags(phrase)
        new_dict[new_str] = chunk_freq_dict[phrase]
    return new_dict


def binom_stoplist(cutoff):
    with open('data/binom.txt', 'r') as f:
        binom_ratios = f.read().decode('utf-8')
    binom_ratios = [l.split('\t') for l in binom_ratios.split('\n') if l]
    stoplist = [word for word, score in binom_ratios if float(score) >= cutoff]
    return stoplist


def stoplist_filter(chunk_freq_dict, stoplist):
    new_dict = {}
    for chunk, freq in chunk_freq_dict.items():
        for word in chunk.split():
            if word in stoplist:
                break
        else:
            new_dict[chunk] = freq
    return new_dict


def build_sorted_chunks(chunk_freq_dict):
    sorted_chunk_dict = defaultdict(list)
    for phrs in chunk_freq_dict.items():
        sorted_chunk_dict[len(phrs[0].split())].append(phrs)
    for num_words in sorted_chunk_dict.keys():
        sorted_chunk_dict[num_words] = sorted(sorted_chunk_dict[num_words],
                                              key=lambda item: item[1],
                                              reverse=True)
    return sorted_chunk_dict


def calc_cvalue(sorted_phrase_dict, min_cvalue):
    cvalue_dict = {}
    triple_dict = {}  # 'candidate string': (f(b), t(b), c(b))
    max_num_words = max(sorted_phrase_dict.keys())

    # Longest candidates.
    for phrs_a, freq_a in sorted_phrase_dict[max_num_words]:
        cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
        if cvalue >= min_cvalue:
            cvalue_dict[phrs_a] = cvalue
            for num_words in reversed(range(1, max_num_words)):
                for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                    if phrs_b in phrs_a:
                        if phrs_b not in triple_dict.keys():  # create triple
                            triple_dict[phrs_b] = (freq_b, freq_a, 1)
                        else:                                 # update triple
                            fb, old_tb, old_cb = triple_dict[phrs_b]
                            triple_dict[phrs_b] = \
                                (fb, old_tb + freq_a, old_cb + 1)

    # Candidates with num. words < max num. words
    num_words_counter = max_num_words - 1
    while num_words_counter > 0:
        for phrs_a, freq_a in sorted_phrase_dict[num_words_counter]:
            if phrs_a not in triple_dict.keys():
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            else:
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * \
                    (freq_a - ((1/triple_dict[phrs_a][2])
                               * triple_dict[phrs_a][1]))
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            if cvalue >= min_cvalue:
                for num_words in reversed(range(1, num_words_counter)):
                    for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                        if phrs_b in phrs_a:
                            if phrs_b not in triple_dict.keys():  # make triple
                                triple_dict[phrs_b] = (freq_b, freq_a, 1)
                            else:                                 # updt triple
                                fb, old_tb, old_cb = triple_dict[phrs_b]
# if/else below: If n(a) is the number of times a has appeared as nested, then
# t(b) will be increased by f(a) - n(a). Frantzi, et al (2000), end of p.5.
                                if phrs_a in triple_dict.keys():
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a -
                                        triple_dict[phrs_a][1], old_cb + 1)
                                else:
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a, old_cb + 1)
        num_words_counter -= 1

    return cvalue_dict


def load_terms():
    with open('corpora/small_domain_terms.txt', 'r') as f:
        ref_raw = f.read().decode('utf-8')
    terms = ref_raw.split('\n')[1:]
    terms = [remove_str_postags(i.strip()) for i in terms]
    return terms


def main(pos_pattern, min_freq, min_cvalue):
    # STEP 1
    domain_sents = load_domain()
    
    # STEP 2
    # Extract matching patterns
    chunks_freqs = chunk_sents(domain_sents, pos_pattern)
    # Remove POS tags from chunks
    chunks_freqs = remove_dict_postags(chunks_freqs)
    # Discard chunks that don't meet minimum frequency
    chunks_freqs = min_freq_filter(chunks_freqs, min_freq)
    # Discard chunks with words in stoplist
    stoplist = binom_stoplist(0.5)
    chunks_freqs = stoplist_filter(chunks_freqs, stoplist)
    # Order candidates first by number of words, then by frequency
    sorted_chunks = build_sorted_chunks(chunks_freqs)

    # STEP 3
    # Calculate C-value
    cvalue_output = calc_cvalue(sorted_chunks, min_cvalue)

    return cvalue_output


if __name__ == '__main__':
    PATTERN = r"""
        TC: {<NC>+<AQ>*(<PDEL><DA>?<NC>+<AQ>*)*}
        """
    MIN_FREQ = 1
    MIN_CVAL = -1000000

    terms = load_terms()
    candidates = main(PATTERN, MIN_FREQ, MIN_CVAL)
    print '[C]', len(candidates)
    print '[T]', len(set(candidates.keys()).intersection(set(terms)))
    print '======'
    precision, recall = evaluation.precision_recall(terms, candidates.keys())
    print '[P]', round(precision, 3)
    print '[R]', round(recall, 3)
    print '======'
    precision_by_segment = evaluation.precision_by_segments(
        terms, candidates.keys(), 4)
    for i, seg_precision in enumerate(precision_by_segment):
        print '[%s] %s' % (i, round(seg_precision, 3))
    recall_list, precision_list = evaluation.precision_at_recall_values(
        terms, candidates.keys())
    evaluation.plot_precision_at_recall_values(recall_list, precision_list)
