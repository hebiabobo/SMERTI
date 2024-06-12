import argparse
import math
import re
import os

import torch
from tqdm import tqdm
import numpy as np
# import language_check
import language_tool_python
from string import punctuation

print('Evaluating number of grammar error:')
tool = language_tool_python.LanguageTool('en-US')


def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string


def sentence_to_words(sentence):
    word_list = sentence.split()
    return word_list


def get_word_list(initial_sentence):
    word_list = sentence_to_words(initial_sentence)
    return word_list


def get_original_len(word_list):
    clean_word_list = []
    for word in word_list:
        if word not in punctuation:
            clean_word_list.append(word)
    original_len = len(clean_word_list)
    return original_len


def get_grammar(path, if_origin):
    input_lines = [line.strip().split('\t') for line in open(path, encoding='utf-8')]
    sum_len = 0
    sum_grammer = 0
    for line in input_lines:
        if if_origin == 1:
            err_num = 0
            for error_type in tool.check(process_string(line[1])):

                if error_type.ruleId == 'UPPERCASE_SENTENCE_START' or error_type.ruleId == 'I_LOWERCASE':  # 忽略句首字母大写和I的大写
                    continue
                else:
                    # print("error_type")
                    # print(error_type)
                    err_num += 1
            grammar_sentence = err_num
            # grammar_sentence = len(tool.check(process_string(line[1])))  # list中的第一个str为RE,第二个str为RE_sentence
            word_list = get_word_list(line[1])
        else:
            err_num = 0
            for error_type in tool.check(process_string(line[0])):
                if error_type.ruleId == 'UPPERCASE_SENTENCE_START' or error_type.ruleId == 'I_LOWERCASE':  # 忽略句首字母大写和I的大写
                    continue
                else:
                    # print(error_type)
                    err_num += 1
            grammar_sentence = err_num
            # grammar_sentence = len(tool.check(process_string(line[0])))  # list中的第一个str为RE,第二个str为RE_sentence
            word_list = get_word_list(line[0])
        original_len = get_original_len(word_list)
        sum_len += original_len
        sum_grammer += grammar_sentence
    avg_grammer = sum_grammer / sum_len
    return avg_grammer


# evalute number of grammar errors
# print('Evaluating number of grammar error:')
# tool = language_check.LanguageTool('en-US')
# tool = language_tool_python.LanguageTool('en-US')
# gramar_err = get_grammer()
# print("number of grammar difference: ", gramar_err)


def choose_dataset(dataset_name, method_name):

    if dataset_name == 'News':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_news.txt")
        method_dir = os.path.join(eval_dir, method_name)
        output_path1 = os.path.join(method_dir, "news_output_20.txt")
        output_path2 = os.path.join(method_dir, "news_output_40.txt")
        output_path3 = os.path.join(method_dir, "news_output_60.txt")
        output_path4 = os.path.join(method_dir, "news_output_80.txt")
    elif dataset_name == 'Amazon':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_reviews.txt")
        method_dir = os.path.join(eval_dir, method_name)
        output_path1 = os.path.join(method_dir, "reviews_output_20.txt")
        output_path2 = os.path.join(method_dir, "reviews_output_40.txt")
        output_path3 = os.path.join(method_dir, "reviews_output_60.txt")
        output_path4 = os.path.join(method_dir, "reviews_output_80.txt")
    elif dataset_name == 'Yelp':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Yelp_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_reviews.txt")
        method_dir = os.path.join(eval_dir, method_name)
        output_path1 = os.path.join(method_dir, "reviews_output_20.txt")
        output_path2 = os.path.join(method_dir, "reviews_output_40.txt")
        output_path3 = os.path.join(method_dir, "reviews_output_60.txt")
        output_path4 = os.path.join(method_dir, "reviews_output_80.txt")
    else:
        print("wrong dataset_name!")
        return False

    grammar_err = get_grammar(original_path, 1)
    print("Average grammar error per token(original): ", grammar_err)
    grammar_err = get_grammar(output_path1, 0)
    print("20:  Average grammar error per token(RE_sentence): : ", grammar_err)
    grammar_err = get_grammar(output_path2, 0)
    print("40:  Average grammar error per token(RE_sentence): : ", grammar_err)
    grammar_err = get_grammar(output_path3, 0)
    print("60:  Average grammar error per token(RE_sentence): : ", grammar_err)
    grammar_err = get_grammar(output_path4, 0)
    print("80:  Average grammar error per token(RE_sentence): : ", grammar_err)


Dataset_name = 'Amazon'
Method_name = 'ste-bart-mutimask-coordinate-similiar'
choose_dataset(Dataset_name, Method_name)
