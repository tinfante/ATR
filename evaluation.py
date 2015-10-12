#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nltk
import matplotlib.pyplot as plt


def precision_recall(terms, candidates):
    term_set = set(terms)
    cand_set = set(candidates)
    precision = nltk.metrics.precision(term_set, cand_set)
    recall = nltk.metrics.recall(term_set, cand_set)
    return precision, recall


def generate_bins(item_list, num_bins):
    start_index = 0
    for bin_num in xrange(num_bins):
        end_index = start_index + len(item_list[bin_num::num_bins])
        yield item_list[start_index:end_index]
        start_index = end_index


def precision_by_segments(all_terms, sorted_candidates, num_bins):
    term_set = set(all_terms)
    segment_precision_list = []
    for segment in generate_bins(sorted_candidates, num_bins):
        segment_set = set(segment)
        segment_precision = nltk.metrics.precision(term_set, segment_set)
        segment_precision_list.append(segment_precision)
    return segment_precision_list


def precision_at_recall_values(all_terms, sorted_candidates):
    term_set = set(all_terms)
    seen_candidates = []
    precision_list = []  # y axis
    recall_list = []  # x axis
    for cand in sorted_candidates:
        seen_candidates.append(cand)
        if cand in term_set:
            seen_set = set(seen_candidates)
            precision = nltk.metrics.precision(term_set, seen_set)
            recall = nltk.metrics.recall(term_set, seen_set)
            precision_list.append(precision)
            recall_list.append(recall)
    return recall_list, precision_list


def plot_precision_at_recall_values(recall_list, precision_list):
    plt.plot(recall_list, precision_list)
    plt.axis([0, 1.01, 0, 1.01])
    plt.xlabel('Cobertura')
    plt.ylabel(u'Precisión')
    plt.show()
