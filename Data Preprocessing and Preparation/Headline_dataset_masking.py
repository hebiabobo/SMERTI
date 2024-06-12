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


path1 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\train_headlines.txt"
path2 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\val_headlines.txt"
path3 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\test_headlines.txt"
path4 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\masked_train_headlines.txt"
path5 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\masked_val_headlines.txt"
path6 = r"D:\PycharmProjects\SMERTI-master\News_Dataset\masked_test_headlines.txt"


mask_train_list = mask_all(path1)
mask_val_list = mask_all(path2)
mask_test_list = mask_all(path3)

# write_file(mask_train_list, path4)
# write_file(mask_val_list, path5)
# write_file(mask_test_list, path6)
