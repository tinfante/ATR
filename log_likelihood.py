#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gelbukh, Alexander, et al. "Automatic term extraction using log-likelihood
based comparison with general reference corpus."
"""

from __future__ import division
from collections import defaultdict
from math import log


def load_corpora(domain_corpus='corpora/big_domain.txt',
                 reference_corpus='corpora/big_reference.txt',
                 accepted_tags=('NC', 'AQ')):

    def read_file(filepath):
        with open(filepath, 'r') as f:
            text = f.read().decode('utf-8')
        return text

    def get_words(corpus, accepted_tags=accepted_tags):
        tokens_lemas = [tl.rsplit('/', 1) for tl in corpus.split()]
        tokens = [tl[0] for tl in tokens_lemas if tl[1] in accepted_tags]
        words = [t.lower() for t in tokens if t.isalnum()]
        return words

    domain_corpus = read_file(domain_corpus)
    reference_corpus = read_file(reference_corpus)
    domain_words = get_words(domain_corpus)
    reference_words = get_words(reference_corpus)

    return domain_words, reference_words


def make_freq_dict(word_list):
    word_freq = defaultdict(int)
    for word in word_list:
        word_freq[word] += 1
    return word_freq


def formula1(domain_words, reference_words):

    corpora_intersection = set(domain_words).intersection(set(reference_words))
    domain_size = len(domain_words)
    reference_size = len(reference_words)
    domain_freq = make_freq_dict(domain_words)
    reference_freq = make_freq_dict(reference_words)

    def formulae2_3():
        domain_expected_freq = {}
        reference_expected_freq = {}
        for word in corpora_intersection:
            formula_division = \
                (domain_freq[word] + reference_freq[word]) \
                / (domain_size + reference_size)
            domain_expected_freq[word] = domain_size * formula_division
            reference_expected_freq[word] = reference_size * formula_division
        return domain_expected_freq, reference_expected_freq

    domain_exp_freq, reference_exp_freq = formulae2_3()

    def formula4(word, score):
        domain_relative_freq = domain_freq[word] / domain_size
        reference_relative_freq = reference_freq[word] / reference_size
        if domain_relative_freq > reference_relative_freq:
            pass
        else:
            score = score * -1
        return score

    scored_words = []
    for word in corpora_intersection:
        domain_parenthesis = domain_freq[word] * \
            log((domain_freq[word] / domain_exp_freq[word]), 2)
        reference_parenthesis = reference_freq[word] * \
            log((reference_freq[word] / reference_exp_freq[word]), 2)
        g_score = 2 * (domain_parenthesis + reference_parenthesis)
        g_score = formula4(word, g_score)
        scored_words.append((g_score, word),)

    return sorted(scored_words, reverse=True)


def save_results(scored_words, filepath):
    with open(filepath, 'a') as f:
        for score, word in scored_words:
            txt_line = '%s\t%.3f\n' % (word, score)
            f.write(txt_line.encode('utf-8'))


if __name__ == '__main__':
    #save_results(formula1(*load_corpora()), 'data/log_likelihood.txt')
