import sys
import argparse
import json
import os
import io
from pprint import pprint
import nltk
import numpy as np
import math
import pkg_resources
from symspellpy import SymSpell, Verbosity
import time
from tqdm import tqdm
from string import punctuation
import re

print("loading symspell dictionary and precalculating edits...")
sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=10)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


# def load_data(path):
#     print("Reading lines...")
#     sentences = []
#     lines = [line.strip().split('\t') for line in open(path, encoding='utf-8')]
#     for line in lines:
#         sentence = line[0]
#         # print("sentence")
#         # print(sentence)
#         sentences.append(sentence)
#     return sentences


def spellcheck(text):
    misspell_count = 0
    my_punctuation = punctuation.replace("'", "")
    # print("text")
    # print(text)
    clean_text = text.translate(str.maketrans('', '', my_punctuation))  # 去除句中标点
    # print("clean_text")
    # print(clean_text)
    clean_text = re.sub("[^a-zA-Z'']+", ' ', clean_text)  # 把句中除了字母以外的任何值变成空格
    # print("clean_text1")
    # print(clean_text)
    clean_text = re.sub(' +', ' ', clean_text)  # 把多个连续空格变成空格
    # print("clean_text2")
    # print(clean_text)
    clean_text = clean_text.strip()  # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    # print("clean_text3")
    # print(clean_text)
    words = clean_text.split()  # word的list
    # print("words")
    # print(words)
    len_text = len(words)
    distance_sum = 0
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=5, include_unknown=True)
        for suggestion in suggestions:
            # print("suggestion")
            # print(suggestion)
            if suggestion.distance > 0:
                distance_sum += suggestion.distance
                misspell_count += 1
    return misspell_count, distance_sum


def evaluate_spelling(path):
    misspell_lst = []
    distance_lst = []
    sentences = []
    lines = [line.strip().split('\t') for line in open(path, encoding='utf-8')]
    for line in lines:
        sentence = line[0]
        misspell_count, distance_sum = spellcheck(sentence)
        misspell_lst.append(misspell_count)
        distance_lst.append(distance_sum)
    if len(misspell_lst) != 0 and len(distance_lst) != 0:
        final_result = [sum(misspell_lst)/len(misspell_lst), sum(distance_lst)/len(distance_lst)]
    return final_result, misspell_lst, distance_lst


eval_dir = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data"
eval_path = os.path.join(eval_dir, "eval_reviews.txt")

path_mutimask = os.path.join(eval_dir, "ste-bart-mutimask-coordinate-similiar")
# filtered_input_path = os.path.join(path_mutimask, "reviews_filtered_input.txt")
path_20 = os.path.join(path_mutimask, "reviews_output_20.txt")

# continuation_file = sys.argv[1]
overall_results = {}
# eval_sentences_list = load_data(path_20)
overall_results['spellcheck'], misspell_lst, distance_lst = evaluate_spelling(path_20)

pprint(overall_results)
# out_filename = continuation_file + '_spellcheck'
# print("Writing spellcheck results to file: ", out_filename)
# with open(out_filename, "w") as fout:
#     pprint(overall_results, stream=fout)
# print("Spellcheck results written to file: ", out_filename)
#
# out_filename_misspell = continuation_file + '_spellcheck_list_misspell'
# out_filename_distance = continuation_file + '_spellcheck_list_distance'
# print("Writing individual spellcheck results to files")
# with open(out_filename_misspell, "w") as fout_misspell:
#     fout_misspell.write('\n'.join([str(x) for x in misspell_lst]))
# with open(out_filename_distance, "w") as fout_distance:
#     fout_distance.write('\n'.join([str(x) for x in distance_lst]))
# print("Individual spellcheck results written to files")
