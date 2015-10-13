#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
- Nazar, "A statistical approach to term extraction", 2011.
- Nazar, Cabré, "Un experimento de extracción de terminología utilizando
  algoritmos estadísticos supervisados", 2011.
- Nazar, Cabré, "Supervised Learning Algorithms Applied to Terminology
  Extraction", 2012.
"""

from __future__ import division
from collections import defaultdict
import random
import nltk
import evaluation


def load_terms():
    with open('corpora/small_domain_terms.txt', 'r') as tf:
        terms_raw = tf.read().decode('utf-8')
    tagged_terms = [l.strip() for l in terms_raw.split('\n')]
    return tagged_terms[1:]


def get_pos_seq_freq(tagged_terms):
    pos_seq_freq_dict = defaultdict(int)
    for tagged_term in tagged_terms:
        pos_seq = ' '.join(w.rsplit('/', 1)[1] for w in tagged_term.split())
        pos_seq_freq_dict[pos_seq] += 1
    return pos_seq_freq_dict


def get_lemma_freq(tagged_sents):
    lemma_freq_dict = defaultdict(int)
    for sent in tagged_sents:
        sent_div = sent.split()
        for word in sent_div:
            word_div = word.rsplit('/', 1)
            if word_div[0].isalnum():
                lemma_freq_dict[word_div[0].lower()] += 1
    return lemma_freq_dict


def get_affix_freq(unique_token_list, affix_type, affix_len):
    affix_freq_dict = defaultdict(int)

    for word in unique_token_list:
        if word.isalnum() and len(word) > affix_len:
            if affix_type == 'prefix':
                affix = word[:affix_len]
            elif affix_type == 'suffix':
                affix = word[-affix_len:]
            affix_freq_dict[affix] += 1
    return affix_freq_dict


def make_term_model(tagged_terms):
    terms = tagged_terms

    # Syntactical part
    pos_seq_freq = get_pos_seq_freq(terms)

    # Lexical part
    lemma_freq = get_lemma_freq(terms)

    # Morphological part
    lemma_f3 = get_affix_freq(lemma_freq.keys(), 'prefix', 3)
    lemma_f4 = get_affix_freq(lemma_freq.keys(), 'prefix', 4)
    lemma_f5 = get_affix_freq(lemma_freq.keys(), 'prefix', 5)
    lemma_l3 = get_affix_freq(lemma_freq.keys(), 'suffix', 3)
    lemma_l4 = get_affix_freq(lemma_freq.keys(), 'suffix', 4)
    lemma_l5 = get_affix_freq(lemma_freq.keys(), 'suffix', 5)

    return {'pos_freq': pos_seq_freq, 'lemma_freq': lemma_freq,
            'lemma_f3': lemma_f3, 'lemma_f4': lemma_f4, 'lemma_f5': lemma_f5,
            'lemma_l3': lemma_l3, 'lemma_l4': lemma_l4, 'lemma_l5': lemma_l5}


def load_general():
    with open('corpora/big_reference.txt', 'r') as gcf:
        ref_raw = gcf.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in ref_raw.split('./Fp')]
    return tagged_sents


def make_general_model(tagged_general):
    # (POS tags not used, but expected as input)
    gen_corp = tagged_general

    # Doesn't use syntactical info

    # Lexical part
    lemma_freq = get_lemma_freq(gen_corp)

    # Morphological part
    lemma_f3 = get_affix_freq(lemma_freq.keys(), 'prefix', 3)
    lemma_f4 = get_affix_freq(lemma_freq.keys(), 'prefix', 4)
    lemma_f5 = get_affix_freq(lemma_freq.keys(), 'prefix', 5)
    lemma_l3 = get_affix_freq(lemma_freq.keys(), 'suffix', 3)
    lemma_l4 = get_affix_freq(lemma_freq.keys(), 'suffix', 4)
    lemma_l5 = get_affix_freq(lemma_freq.keys(), 'suffix', 5)

    return {'lemma_freq': lemma_freq,
            'lemma_f3': lemma_f3, 'lemma_f4': lemma_f4, 'lemma_f5': lemma_f5,
            'lemma_l3': lemma_l3, 'lemma_l4': lemma_l4, 'lemma_l5': lemma_l5}


def load_analysis():
    with open('corpora/small_domain.txt', 'r') as corpf:  # small corpus
        corp = corpf.read().decode('utf-8')
    tagged_sents = [s.strip()+' ./Fp' for s in corp.split('./Fp')]
    tagged_sents = [s.split() for s in tagged_sents]
    return tagged_sents


def chunk_sents(pos_sequence, tagged_sents):
    grammar = r'TC: {%s}' % ''.join(['<%s>' % t for t in pos_sequence.split()])
    chunker = nltk.RegexpParser(grammar)
    chunks = []
    for sent in tagged_sents:
        lemtagdiv_sent = [tuple(w.rsplit('/', 1)) for w in sent]
        for chnk in chunker.parse(lemtagdiv_sent).subtrees():
            if str(chnk).startswith('(TC'):
                phrase = chnk.__unicode__()[4:-1]
                if '\n' in phrase:
                    phrase = ' '.join(phrase.split())
                chunks.append(phrase)
    return chunks


def calc_lexical_coef(candidate, term_model, gen_model, s=0.001,
                      stoplist=['del']):

    term_lemma_num = sum(term_model['lemma_freq'].values())
    gen_lemma_sum = sum(gen_model['lemma_freq'].values())

    lemma_score_list = []
    lemmas = [w.rsplit('/', 1)[0] for w in candidate.split()]
    for lem in lemmas:
        if lem in stoplist:  # TODO: if lem in stoplist; continue statement
            continue
        relative_lem_freq_in_terms = 0.0
        if lem in term_model['lemma_freq'].keys():
            relative_lem_freq_in_terms = \
                term_model['lemma_freq'][lem] / term_lemma_num

        relative_lem_freq_in_gen = 0.0
        if lem in gen_model['lemma_freq'].keys():
            relative_lem_freq_in_gen = \
                gen_model['lemma_freq'][lem] / gen_lemma_sum

        lemma_score = \
            relative_lem_freq_in_terms / (relative_lem_freq_in_gen + s)
        lemma_score_list.append(lemma_score)
    lemma_coef = sum(lemma_score_list) / len(lemma_score_list)

    lexical_coef = lemma_coef  # Falta la parte de palabras.

    return lexical_coef


def calc_morph_coef(candidate, term_model, gen_model, s=0.001,
                    stoplist=['del']):

    lem_f3_score_list = []
    term_lf3_num = sum(term_model['lemma_f3'].values())
    gen_lf3_num = sum(gen_model['lemma_f3'].values())

    lem_f4_score_list = []
    term_lf4_num = sum(term_model['lemma_f4'].values())
    gen_lf4_num = sum(gen_model['lemma_f4'].values())

    lem_f5_score_list = []
    term_lf5_num = sum(term_model['lemma_f5'].values())
    gen_lf5_num = sum(gen_model['lemma_f5'].values())

    lem_l3_score_list = []
    term_ll3_num = sum(term_model['lemma_l3'].values())
    gen_ll3_num = sum(gen_model['lemma_l3'].values())

    lem_l4_score_list = []
    term_ll4_num = sum(term_model['lemma_l4'].values())
    gen_ll4_num = sum(gen_model['lemma_l4'].values())

    lem_l5_score_list = []
    term_ll5_num = sum(term_model['lemma_l5'].values())
    gen_ll5_num = sum(gen_model['lemma_l5'].values())

    lemmas = [w.rsplit('/', 1)[0] for w in candidate.split()]
    for lem in lemmas:
        if lem in stoplist:
            continue

        lf3_affix = lem[:3]
        relative_lf3_freq_in_terms = 0.0
        if len(lf3_affix) == 3 and lf3_affix in term_model['lemma_f3'].keys():
            relative_lf3_freq_in_terms = \
                term_model['lemma_f3'][lf3_affix] / term_lf3_num
        relative_lf3_freq_in_gen = 0.0
        if len(lf3_affix) == 3 and lf3_affix in gen_model['lemma_f3'].keys():
            relative_lf3_freq_in_gen = \
                gen_model['lemma_f3'][lf3_affix] / gen_lf3_num
        lf3_score = \
            relative_lf3_freq_in_terms / (relative_lf3_freq_in_gen + s)
        lem_f3_score_list.append(lf3_score)

        lf4_affix = lem[:4]
        relative_lf4_freq_in_terms = 0.0
        if len(lf4_affix) == 4 and lf4_affix in term_model['lemma_f4'].keys():
            relative_lf4_freq_in_terms = \
                term_model['lemma_f4'][lf4_affix] / term_lf4_num
        relative_lf4_freq_in_gen = 0.0
        if len(lf4_affix) == 4 and lf4_affix in gen_model['lemma_f4'].keys():
            relative_lf4_freq_in_gen = \
                gen_model['lemma_f4'][lf4_affix] / gen_lf4_num
        lf4_score = \
            relative_lf4_freq_in_terms / (relative_lf4_freq_in_gen + s)
        lem_f4_score_list.append(lf4_score)

        lf5_affix = lem[:5]
        relative_lf5_freq_in_terms = 0.0
        if len(lf5_affix) == 5 and lf5_affix in term_model['lemma_f5'].keys():
            relative_lf5_freq_in_terms = \
                term_model['lemma_f5'][lf5_affix] / term_lf5_num
        relative_lf5_freq_in_gen = 0.0
        if len(lf5_affix) == 5 and lf5_affix in gen_model['lemma_f5'].keys():
            relative_lf5_freq_in_gen = \
                gen_model['lemma_f5'][lf5_affix] / gen_lf5_num
        lf5_score = \
            relative_lf5_freq_in_terms / (relative_lf5_freq_in_gen + s)
        lem_f5_score_list.append(lf5_score)

        ll3_affix = lem[-3:]
        relative_ll3_freq_in_terms = 0.0
        if len(ll3_affix) == 3 and ll3_affix in term_model['lemma_l3'].keys():
            relative_ll3_freq_in_terms = \
                term_model['lemma_l3'][ll3_affix] / term_ll3_num
        relative_ll3_freq_in_gen = 0.0
        if len(ll3_affix) == 3 and ll3_affix in gen_model['lemma_l3'].keys():
            relative_ll3_freq_in_gen = \
                gen_model['lemma_l3'][ll3_affix] / gen_ll3_num
        ll3_score = \
            relative_ll3_freq_in_terms / (relative_ll3_freq_in_gen + s)
        lem_l3_score_list.append(ll3_score)

        ll4_affix = lem[-4:]
        relative_ll4_freq_in_terms = 0.0
        if len(ll4_affix) == 4 and ll4_affix in term_model['lemma_l4'].keys():
            relative_ll4_freq_in_terms = \
                term_model['lemma_l4'][ll4_affix] / term_ll4_num
        relative_ll4_freq_in_gen = 0.0
        if len(ll4_affix) == 4 and ll4_affix in gen_model['lemma_l4'].keys():
            relative_ll4_freq_in_gen = \
                gen_model['lemma_l4'][ll4_affix] / gen_ll4_num
        ll4_score = \
            relative_ll4_freq_in_terms / (relative_ll4_freq_in_gen + s)
        lem_l4_score_list.append(ll4_score)

        ll5_affix = lem[-5:]
        relative_ll5_freq_in_terms = 0.0
        if len(ll5_affix) == 5 and ll5_affix in term_model['lemma_l5'].keys():
            relative_ll5_freq_in_terms = \
                term_model['lemma_l5'][ll5_affix] / term_ll5_num
        relative_ll5_freq_in_gen = 0.0
        if len(ll5_affix) == 5 and ll5_affix in gen_model['lemma_l5'].keys():
            relative_ll5_freq_in_gen = \
                gen_model['lemma_l5'][ll5_affix] / gen_ll5_num
        ll5_score = \
            relative_ll5_freq_in_terms / (relative_ll5_freq_in_gen + s)
        lem_l5_score_list.append(ll5_score)

    lf3_coef = sum(lem_f3_score_list) / len(lem_f3_score_list)
    lf4_coef = sum(lem_f4_score_list) / len(lem_f4_score_list)
    lf5_coef = sum(lem_f5_score_list) / len(lem_f5_score_list)
    ll3_coef = sum(lem_l3_score_list) / len(lem_l3_score_list)
    ll4_coef = sum(lem_l4_score_list) / len(lem_l4_score_list)
    ll5_coef = sum(lem_l5_score_list) / len(lem_l5_score_list)

    morph_coef = (lf3_coef + lf4_coef + lf5_coef +
                  ll3_coef + ll4_coef + ll5_coef) / 6
    return morph_coef


def calc_syntactic_coef(pos_pattern, term_model):
    syn_coef = term_model['pos_freq'][pos_pattern] / \
        sum(term_model['pos_freq'].values())
    return syn_coef


def remove_str_postags(tagged_str):
    stripped_str = ' '.join([w.split('/')[0] for w in tagged_str.split()])
    return stripped_str


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


def stoplist_filter(sorted_candidates, stoplist):
    new_sorted_candidates = []
    for score, candidate in sorted_candidates:
        for word in candidate.split():
            if word in stoplist:
                break
        else:
            new_sorted_candidates.append((candidate, score),)

    return new_sorted_candidates


def split_train_test(domain_sents, terms):
    random.shuffle(domain_sents)

    train_corpus = domain_sents[:int(0.5 * len(domain_sents))]
    train_terms = []
    for sent in train_corpus:
        for term in terms:
            if term in ' '.join(sent):
                train_terms.append(term)
    train_terms = [t for t in train_terms if t]

    test_corpus = domain_sents[int(0.5 * len(domain_sents)):]
    test_terms = []
    for sent in test_corpus:
        for term in terms:
            if term in ' '.join(sent):
                test_terms.append(term)
    test_terms = [remove_str_postags(t) for t in test_terms if t]

    return train_terms, test_corpus, test_terms


def main(train_terms, test_corpus):
    term_model = make_term_model(train_terms)

    general_sents = load_general()
    general_model = make_general_model(general_sents)

    candidate_scores = []
    pos_patterns = term_model['pos_freq'].keys()
    for pos_seq in pos_patterns:
        syn_coef = calc_syntactic_coef(pos_seq, term_model)
        chunks = chunk_sents(pos_seq, test_corpus)
        chunk_freq_dict = defaultdict(int)
        for chnk in chunks:
            chunk_freq_dict[chnk] += 1

        for candidate in chunk_freq_dict.keys():
            cand_freq = chunk_freq_dict[candidate]
            lex_coef = calc_lexical_coef(candidate, term_model, general_model)
            morph_coef = calc_morph_coef(candidate, term_model, general_model)
            candidate_coef = cand_freq * syn_coef * lex_coef * morph_coef
            candidate_scores.append((candidate_coef, candidate),)
    candidate_scores = sorted(candidate_scores, reverse=True)
    candidate_scores = [(score, remove_str_postags(cand)) for score, cand
                        in candidate_scores]

    stoplist = binom_stoplist(0.5)  # 0.5 buen valor
    #stoplist = log_likelihood_stoplist(5)
    print len(candidate_scores)
    candidate_scores = stoplist_filter(candidate_scores, stoplist)
    print len(candidate_scores)

    return candidate_scores


if __name__ == '__main__':
    terms = load_terms()

    domain_sents = load_analysis()
    train_terms, test_corpus, test_terms = \
        split_train_test(domain_sents, terms)

    candidates = main(train_terms, test_corpus)
    candidates = [word for word, score in candidates]

    print '[C]', len(candidates)
    print '[T]', len(set(candidates).intersection(set(test_terms)))
    print '======'
    precision, recall = evaluation.precision_recall(test_terms, candidates)
    print '[P]', round(precision, 3)
    print '[R]', round(recall, 3)
    print '======'
    precision_by_segment = evaluation.precision_by_segments(
        test_terms, candidates, 4)
    for i, seg_precision in enumerate(precision_by_segment):
        print '[%s] %s' % (i, round(seg_precision, 3))
    recall_list, precision_list = evaluation.precision_at_recall_values(
        test_terms, candidates)
    evaluation.plot_precision_at_recall_values(recall_list, precision_list)
