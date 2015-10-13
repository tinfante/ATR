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
    stoplist = \
        [word for word, score in binom_ratios if float(score) >= cutoff]
    return stoplist


def log_likelihood_stoplist(cutoff):
    with open('data/log_likelihood.txt', 'r') as f:
        loglike_ratios = f.read().decode('utf-8')
    loglike_ratios = [l.split('\t') for l in loglike_ratios.split('\n') if l]
    stoplist = \
        [word for word, score in loglike_ratios if float(score) <= cutoff]
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


def make_contextword_weight_dict(real_term_list, tagged_sents, valid_tags,
                                 context_size):
    context_word_dict = defaultdict(int)
    num_terms_seen = 0
    for term in real_term_list:
        for sent in tagged_sents:
            sent_str = ' '.join(w[0] for w in sent)
            if term in sent_str:
                term_split = term.split()
                for wt_idx in range(len(sent) - len(term_split)):
                    # wt_idx = wordtag_index.
                    word_size_window = [
                        w[0] for w in
                        sent[wt_idx:wt_idx+len(term_split)]]
                    if term_split == word_size_window:
                        left_context = sent[:wt_idx][-context_size:]
                        right_context = \
                            sent[wt_idx+len(term_split):][:context_size]
                        context = left_context + right_context
                        valid_words = [w[0] for w in context if
                                       w[1] in valid_tags]
                        for word in valid_words:
                            context_word_dict[word] += 1
                        num_terms_seen += 1
                        break  #  1 term match per sentence
    context_word_dict = dict(  # Transform keys: freqs -> weights
        (k, v/num_terms_seen) for k, v in context_word_dict.items())
    return context_word_dict


def calc_ncvalue(cvalue_results, tagged_sents, contextword_weight_dict,
                 valid_tags, context_size):
    ncvalue_dict = {}
    for candidate, cand_cvalue in cvalue_results.items():
        ccw_freq_dict = defaultdict(int)  # ccw = candidate_context_words
        for sent in tagged_sents:
            sent_str = ' '.join(w[0] for w in sent)
            if candidate in sent_str:
                candidate_split = candidate.split()
                for wt_idx in range(len(sent) - len(candidate_split)):
                    word_size_window = [
                        w[0] for w in
                        sent[wt_idx:wt_idx+len(candidate_split)]]
                    if candidate_split == word_size_window:
                        left_context = sent[:wt_idx][-context_size:]
                        right_context = \
                            sent[wt_idx+len(candidate_split):][:context_size]
                        # TODO: see same bit in previous function.
                        context = left_context + right_context
                        valid_words = [w[0] for w in context if
                                       w[1].lower() in valid_tags]
                        for word in valid_words:
                            ccw_freq_dict[word] += 1
                        break  # 1 candidate match per sentence
        context_factors = []
        for word in ccw_freq_dict.keys():
            if word in contextword_weight_dict.keys():
                context_factors.append(
                    ccw_freq_dict[word] * contextword_weight_dict[word])
        ncvalue = (0.8 * cand_cvalue) + (0.2 * sum(context_factors))
        ncvalue_dict[candidate] = ncvalue
    return ncvalue_dict


def load_terms():
    with open('corpora/small_domain_terms.txt', 'r') as f:
        ref_raw = f.read().decode('utf-8')
    terms = ref_raw.split('\n')[1:]
    terms = [remove_str_postags(i.strip()) for i in terms]
    return terms


def main(domain_corpus, pos_pattern, min_freq, min_cvalue):
    # STEP 1
    domain_sents = domain_corpus

    # STEP 2
    # Extract matching patterns
    chunks_freqs = chunk_sents(domain_sents, pos_pattern)

    # Remove POS tags from chunks
    chunks_freqs = remove_dict_postags(chunks_freqs)

    # Discard chunks that don't meet minimum frequency
    chunks_freqs = min_freq_filter(chunks_freqs, min_freq)

    # Discard chunks with words in stoplist
    stoplist = binom_stoplist(0.5)  # 0.5 da buenos resultados
    #stoplist = log_likelihood_stoplist(400)
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
    domain_corpus = load_domain()
    candidates = main(domain_corpus, PATTERN, MIN_FREQ, MIN_CVAL)
    sorted_candidates = [cand for cand, score in sorted(
        candidates.items(), key=lambda x: x[1], reverse=True)]
    print '\nC-VALUE'
    print '========'
    print '[C]', len(sorted_candidates)
    print '[T]', len(set(sorted_candidates).intersection(set(terms)))
    print '========'
    precision, recall = evaluation.precision_recall(terms, sorted_candidates)
    print '[P]', round(precision, 3)
    print '[R]', round(recall, 3)
    print '========'
    precision_by_segment = evaluation.precision_by_segments(
        terms, sorted_candidates, 4)
    for i, seg_precision in enumerate(precision_by_segment):
        print '[%s] %s' % (i, round(seg_precision, 3))
    recall_list, precision_list = evaluation.precision_at_recall_values(
        terms, sorted_candidates)
    evaluation.plot_precision_at_recall_values(recall_list, precision_list)

    cvalue_top = [c for c in sorted_candidates[:int(len(candidates) * 0.2)]]
    context_words = make_contextword_weight_dict(
        cvalue_top, domain_corpus, ['NC', 'AQ', 'VM'], 5)
    ncvalue_output = calc_ncvalue(
        candidates, domain_corpus, context_words, ['NC', 'AQ', 'VM'], 5)
    sorted_ncvalue = [cand for cand, score in sorted(
        ncvalue_output.items(), key=lambda x: x[1], reverse=True)]
    precision, recall = \
        evaluation.precision_recall(terms, sorted_ncvalue)
    print '\n\nNC-VALUE'
    print '========'
    precision_by_segment = evaluation.precision_by_segments(
        terms, sorted_ncvalue, 4)
    for i, seg_precision in enumerate(precision_by_segment):
        print '[%s] %s' % (i, round(seg_precision, 3))
    recall_list, precision_list = evaluation.precision_at_recall_values(
        terms, sorted_ncvalue)
    evaluation.plot_precision_at_recall_values(recall_list, precision_list)
