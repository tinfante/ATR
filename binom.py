#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from collections import defaultdict
from scipy.stats import binom


def load_domain():
    with open('corpora/big_domain.txt', 'r') as f:
        dom_raw = f.read().decode('utf-8')
    dom_freq_dict = defaultdict(int)
    for word in dom_raw.split():
        if word.split('/')[1] in ('NC', 'AQ'):
            dom_freq_dict[word.rsplit('/', 1)[0].lower()] += 1
    return dom_freq_dict


def load_reference():
    with open('corpora/big_reference.txt', 'r') as f:
        ref_raw = f.read().decode('utf-8')
    ref_freq_dict = defaultdict(int)
    for word in ref_raw.split():
        if word.split('/')[1] in ('NC', 'AQ'):
            ref_freq_dict[word.rsplit('/', 1)[0].lower()] += 1
    return ref_freq_dict


def binom_prob_dict(lematag_freq_dict):
    lematag_binom_prob_dict = {}
    num_words = sum(lematag_freq_dict.values())
    for word in lematag_freq_dict.keys():
        word_freq = lematag_freq_dict[word]
        word_prob = word_freq / num_words
        word_binom_prob = round(binom.pmf(word_freq, num_words, word_prob), 5)
        lematag_binom_prob_dict[word] = word_binom_prob
    return lematag_binom_prob_dict


def sorted_freq_dict(freq_dict):
    return sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)


def binom_ratio(dom_binom_dict, ref_binom_dict):
    shared = []
    key_intersect = \
        set(dom_binom_dict.keys()).intersection(set(ref_binom_dict.keys()))
    for lempos in key_intersect:
        shared.append(
            (lempos, round(dom_binom_dict[lempos]/ref_binom_dict[lempos], 3)),)
    return dict(shared)


def write_results(binom_sorted_list):
    with open('data/binom.txt', 'a') as f:
        for word, score in binom_sorted_list:
            newline = '%s\t%.3f\n' % (word, score)
            f.write(newline.encode('utf-8'))


def main():
    dom_freq = load_domain()
    ref_freq = load_reference()
    dom_binom = binom_prob_dict(dom_freq)
    ref_binom = binom_prob_dict(ref_freq)
    brat_dict = binom_ratio(dom_binom, ref_binom)
    brat_list = sorted_freq_dict(brat_dict)
    return brat_list


if __name__ == '__main__':
    #write_results(main())
