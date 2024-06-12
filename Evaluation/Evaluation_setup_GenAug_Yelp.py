#!/usr/bin/env python
# coding: utf-8


import io, json, os, collections, pprint, time
import re

import random
import operator

from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'E:\stanfordcorenlp\stanford-corenlp-full-2018-02-27')


def extract_leaves(tree):
    # print("---extract_leaves---")
    leaves_list = []
    # print("tree")
    # print(tree)
    # print("tree.subtrees()")
    # print(tree.subtrees())
    # print("tree.leaves()")
    # print(tree.leaves())
    for i in tree.subtrees():
        # print("i")
        # print(i)
        if i.label() in ["NN", "NNS", "NP"]:  # 如果是名词单复数、名词短语(NP)
            # 并不只是抽出了单个token，而且抽出了所有的命名实体(包括短语),token是leave，短语是子树
            leaves_list.append(i.leaves())  # 就保存在leaves_list中
    return leaves_list


def create_entities_list(list_of_lists):
    entities_list = []
    for entity_list in list_of_lists:
        entity = ' '.join(entity_list).replace(" '", "'")
        if entity not in entities_list:
            entities_list.append(entity)
    return entities_list


def get_trees(initial_sentence):
    # print("---get_trees---")
    full_leaves_list = []
    sentences = re.split('[?.!]', initial_sentence)
    sentences = list(filter(lambda x: x not in ['', ' '], sentences))
    # print("sentences")
    # print(sentences)
    for sentence in sentences:
        sentence = Tree.fromstring(nlp.parse(sentence))
        leaves = extract_leaves(sentence)  # 返回 词性是名词单复数、名词短语(NP) 的命名实体(包含token和短语)
        full_leaves_list = full_leaves_list + leaves  # full_leaves_list:句子sentences中所有的 名词单复数、名词短语(NP)
    # print("full_leaves_list")  # 格式:[[''],[''],...,['']]
    # print(full_leaves_list)
    clean_leaves_list = create_entities_list(full_leaves_list)  # 换格式
    # print("clean_leaves_list")  # 格式:['','',...,'']
    # print(clean_leaves_list)
    return clean_leaves_list


def get_lines(path):
    lines_list = []
    file = io.open(path, encoding='utf-8')
    for line in file:
        # print("line")
        # print(line)
        clean_line = re.sub('\s+', ' ', line).strip()  # \s是匹配所有空白符
        # print("clean_line")
        # print(clean_line)
        lines_list.append(clean_line)
    return lines_list


def get_sent_lines(path, sent):
    sent_lines = []
    file = io.open(path, encoding='utf-8')
    for line in file:
        clean_line = re.sub('\s+', ' ', line).strip()
        sent_line = [clean_line, sent]
        sent_lines.append(sent_line)
    return sent_lines


def get_nouns(full_list):
    nouns_dict = {}
    counter = 0
    for line in full_list:
        if counter % 1000 == 0:
            print("Got nouns for {} lines.".format(counter))
            # print("line")
            # print(line)
        line_nouns = get_trees(line)  # 获取 句子sentences中所有的 名词单复数、名词短语(NP)
        # if counter % 1000 == 0:
            # print("line_nouns")
            # print(line_nouns)
        for noun in line_nouns:  # 统计 各个 句子sentences中所有的 名词单复数、名词短语(NP) 的个数。 字典的value是个数
            if noun in nouns_dict:
                nouns_dict[noun] += 1
            else:
                nouns_dict[noun] = 1
        counter += 1
    return nouns_dict  # 各个 句子sentences中所有的 名词单复数、名词短语(NP) 的个数。 字典的value是个数


def get_sorted_list(nouns_dict):
    sorted_list = sorted(nouns_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list


def get_test_lines(pos_list, neg_list, test_nouns):
    temp_test_lines_1 = {}
    temp_test_lines_2 = {}
    temp_test_lines_3 = {}

    final_test_lines_1 = []
    final_test_lines_2 = []
    final_test_lines_3 = []

    for noun in test_nouns:
        noun_lines_pos = list(filter(lambda x: noun not in x[0] and len(x[0].split()) > 5, pos_list))
        noun_lines_neg = list(filter(lambda x: noun not in x[0] and len(x[0].split()) > 5, neg_list))

        random.seed(3)
        random.shuffle(noun_lines_pos)
        random.shuffle(noun_lines_neg)

        noun_lines_pos_1 = noun_lines_pos[:50]
        noun_lines_neg_1 = noun_lines_neg[:50]
        noun_lines_full_1 = noun_lines_pos_1 + noun_lines_neg_1

        noun_lines_pos_2 = noun_lines_pos[50:100]
        noun_lines_neg_2 = noun_lines_neg[50:100]
        noun_lines_full_2 = noun_lines_pos_2 + noun_lines_neg_2

        noun_lines_pos_3 = noun_lines_pos[100:150]
        noun_lines_neg_3 = noun_lines_neg[100:150]
        noun_lines_full_3 = noun_lines_pos_3 + noun_lines_neg_3

        temp_test_lines_1[noun] = noun_lines_full_1
        temp_test_lines_2[noun] = noun_lines_full_2
        temp_test_lines_3[noun] = noun_lines_full_3

    for noun in test_nouns:
        for line in temp_test_lines_1[noun]:
            final_test_lines_1.append([noun] + line)
        for line in temp_test_lines_2[noun]:
            final_test_lines_2.append([noun] + line)
        for line in temp_test_lines_3[noun]:
            final_test_lines_3.append([noun] + line)

    return final_test_lines_1, final_test_lines_2, final_test_lines_3


def write_nouns(nouns_list, path):
    f = io.open(path, "w", encoding='utf-8')
    print("Currently writing nouns to file ...")
    f.write('\n'.join('{}\t{}'.format(x[0], x[1]) for x in nouns_list))
    f.close()
    print("Nouns successfully written to file!")


def write_eval_lines(final_test_lines, path):
    f = io.open(path, "w", encoding='utf-8')
    print("Currently writing evaluation lines to file ...")
    f.write('\n'.join('{}\t{}\t{}'.format(x[0], x[1], x[2]) for x in final_test_lines))
    f.close()
    print("Evaluation lines successfully written to file!")


main_path = r"E:\PycharmProjects\SMERTI-master"
corpus_name = "Yelp_Dataset"
corpus = os.path.join(main_path, corpus_name)
data_path = corpus
eval_path = os.path.join(corpus, r"evaluation_data")

formatted_file = os.path.join(data_path, "mask_final_train_reviews.txt")
formatted_file_valid = os.path.join(data_path, "mask_final_valid_reviews.txt")

path_full = os.path.join(data_path, "final_test_reviews.txt")

path_eval = os.path.join(eval_path, "yelp_train_p1_SMERTI.txt")

# path_nouns = os.path.join(eval_path, "full_nouns_list.txt")


full_list = get_lines(path_full)  # 句子
nouns_dict = get_nouns(full_list)  # 去重后的 句子sentences中所有的 名词单复数、名词短语(NP)
nouns_list = get_sorted_list(nouns_dict)  # most frequent 排序?
# write_nouns(nouns_list, path_nouns)

frequent_nouns = nouns_list[:int(round(len(nouns_list) / 10))]  # Get top 10% most frequent nouns
print("Frequent nouns (top 10%) (look at first 30): ", frequent_nouns[:30])

# Manually selected test nouns (from frequent_nouns)
test_nouns = ['food', 'service', 'place', 'staff', 'time', 'customer', 'atmosphere', 'pizza', 'restaurant',
              'chicken']  # for Yelp dataset


final_test_lines = get_test_lines(full_list, test_nouns)

write_eval_lines(final_test_lines, path_eval)

