# -*- coding: utf-8 -*-
import io, json, os, collections, pprint, time
import re
from string import punctuation
from random import sample
import math
import unicodedata


def mask_all(path):
    mask_list = []
    f = io.open(path, encoding = 'utf-8')
    lines = f.readlines()
    # print(lines)
    total = len(lines)
    # print(total)
    counter = 0
    print("Currently masking lines ...")
    for l in lines:
        # print(l)
        if 0 <= counter <= int(round(total / 3)):
            masked_text = mask_text(l, 0.15)
        elif int(round(total / 3)) < counter <= int(round((total * 2) / 3)):
            masked_text = mask_text(l, 0.30)
        else:
            masked_text = mask_text(l, 0.45)
        ##For transformer:
        mask_list.append(masked_text + '\n')
        ##For RNN:
        #mask_list.append(masked_text + '\t' + l)
        counter += 1
    return mask_list


def mask_text(line, value):
    word_list = (line.rstrip()).split()  # 把句子转换为word的list
    # print(word_list)
    num = int(round(len(word_list) * value))  # num:mask的token数量
    # print(num)
    mask_locs = set(sample(range(len(word_list)), num))  # sample:从序列range(len(word_list))中随机抽取num个元素，并将num个元素list形式返回
    # print(set(sample(range(len(word_list)), num)))
    masked = list(('[mask]' if i in mask_locs and word_list[i] not in punctuation else c for i,c in enumerate(word_list)))  # 添加[mask]; i是index,c是word
    # print(masked)
    masked_groups = mask_groupings(masked)  # 合并相邻的[mask]
    # print(masked_groups)
    masked_text = ' '.join(masked_groups)  # 把list还原为句子
    # print(masked_text)
    return masked_text


def mask_groupings(masked_list):  # 合并相邻的[mask]
    masked_group_list = []
    previous_element = ""
    for element in masked_list:
        if element != "[mask]":
            masked_group_list.append(element)
        elif element == "[mask]":
            if element != previous_element:
                masked_group_list.append(element)
        previous_element = element  # 暂时保存上一个word(或mask)
    return masked_group_list


def write_file(lst, path):
    f = io.open(path, "w", encoding = 'utf-8')
    print("Currently writing lines to file ...")
    f.writelines(lst)
    f.close()
    print("Lines successfully written to file!")


path1 = r"E:\PycharmProjects\GenAug-master\data\yelp_train.txt"
path4 = r"E:\PycharmProjects\GenAug-master\data\yelp_val.txt"
path7 = r"E:\PycharmProjects\GenAug-master\data\yelp_test.txt"

path22 = r"E:\PycharmProjects\GenAug-master\data\mask_final_train_reviews.txt"
path23 = r"E:\PycharmProjects\GenAug-master\data\mask_final_valid_reviews.txt"
# path24 = r"E:\PycharmProjects\GenAug-master\data\mask_final_test_reviews.txt"


final_mask_train_reviews = mask_all(path1)
final_mask_valid_reviews = mask_all(path4)
final_mask_test_reviews = mask_all(path7)

write_file(final_mask_train_reviews, path22)
write_file(final_mask_valid_reviews, path23)
# write_file(final_mask_test_reviews, path24)
