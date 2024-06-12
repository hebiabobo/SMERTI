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


# In[ ]:


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


# In[ ]:


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


def get_news_lines(full_list, test_nouns):
    temp_news_lines_1 = {}
    temp_news_lines_2 = {}
    temp_news_lines_3 = {}

    final_news_lines_1 = []
    final_news_lines_2 = []
    final_news_lines_3 = []

    for noun in test_nouns:
        noun_lines = list(filter(lambda x: noun not in x and len(x.split()) > 5, full_list))

        random.seed(3)
        random.shuffle(noun_lines)

        noun_lines_1 = noun_lines[:100]
        noun_lines_2 = noun_lines[100:200]
        noun_lines_3 = noun_lines[200:300]

        temp_news_lines_1[noun] = noun_lines_1
        temp_news_lines_2[noun] = noun_lines_2
        temp_news_lines_3[noun] = noun_lines_3

    for noun in test_nouns:
        for line in temp_news_lines_1[noun]:
            final_news_lines_1.append(noun + '\t' + line)
        for line in temp_news_lines_2[noun]:
            final_news_lines_2.append(noun + '\t' + line)
        for line in temp_news_lines_3[noun]:
            final_news_lines_3.append(noun + '\t' + line)

    return final_news_lines_1, final_news_lines_2, final_news_lines_3


def write_nouns(nouns_list, path):
    f = io.open(path, "w", encoding='utf-8')
    print("Currently writing nouns to file ...")
    f.write('\n'.join('{}\t{}'.format(x[0], x[1]) for x in nouns_list))
    f.close()
    print("Nouns successfully written to file!")


def write_news_lines(final_news_lines, path):
    f = io.open(path, "w", encoding='utf-8')
    print("Currently writing news evaluation headlines to file ...")
    f.write('\n'.join(x for x in final_news_lines))
    f.close()
    print("News evaluation headlines successfully written to file!")


main_path = r"E:\PycharmProjects\SMERTI-master"
corpus_name = "News_Dataset"
corpus = os.path.join(main_path, corpus_name)
data_path = corpus
eval_path = os.path.join(corpus, r"evaluation_data")
print(corpus)
print(eval_path)

formatted_file = os.path.join(data_path, "masked_train_headlines.txt")
formatted_file_valid = os.path.join(data_path, "masked_val_headlines.txt")

path_full = os.path.join(data_path, "test_headlines.txt")

path_eval_1 = os.path.join(eval_path, "eval_headlines_(1).txt")
path_eval_2 = os.path.join(eval_path, "eval_headlines_(2).txt")
path_eval_3 = os.path.join(eval_path, "eval_headlines_(3).txt")

path_nouns = os.path.join(eval_path, "full_nouns_list.txt")


full_list = get_lines(path_full)  # 句子
nouns_dict = get_nouns(full_list)  # 去重后的 句子sentences中所有的 名词单复数、名词短语(NP)
nouns_list = get_sorted_list(nouns_dict)  # most frequent 排序?
write_nouns(nouns_list, path_nouns)

frequent_nouns = nouns_list[:int(round(len(nouns_list) / 10))]  # Get top 10% most frequent nouns
print("Frequent nouns (top 10%) (look at first 30): ", frequent_nouns[:30])

# Manually selected test nouns (from frequent_nouns)

test_nouns_news = ['trump', 'photos', 'video', 'world', 'women', 'life', 'kids', 'people', 'week',
                   'obama']  # for news headlines dataset

final_news_lines_1, final_news_lines_2, final_news_lines_3 = get_news_lines(full_list, test_nouns_news)

write_news_lines(final_news_lines_1, path_eval_1)
write_news_lines(final_news_lines_2, path_eval_2)
write_news_lines(final_news_lines_3, path_eval_3)
