#!/usr/bin/env python
# coding: utf-8

# Note: format of the files inputted to these functions as follows:
# 
# -Input_path: each line, when split by tab, contains an RE and original piece of text (either review or headline), one per line
# 
# -Output_path: each line, when split by tab, contains a generated piece of text and the actual masking rate for that generation as the first two elements, one per line


# #Sentiment Scores

# nltk.download('vader_lexicon')

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import corpus_bleu
from flair.embeddings import FlairEmbeddings

analyser = SentimentIntensityAnalyzer()
# # get language model
language_model = FlairEmbeddings('news-forward').lm


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    # print("{:-<40} {}".format(sentence, str(score)))
    return score


def sentiment_helper(line_sent):  # 判断此句是positive or negative or neutral
    if line_sent["compound"] >= 0.05:
        sentiment = "positive"
    elif line_sent["compound"] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return sentiment


def vader_sentiments(input_path, output_path, total_lines):
    input_sentiments_lst = []
    output_sentiments_lst = []
    total_sent_count = 0
    input_lines = [line.strip().split('\t') for line in open(input_path, encoding='utf-8')]
    output_lines = [line.strip().split('\t') for line in open(output_path, encoding='utf-8')]

    for input_line, output_line in zip(input_lines, output_lines):
        input_line_sent = sentiment_analyzer_scores(input_line[1])  # 计算原句的情感极性分数
        output_line_sent = sentiment_analyzer_scores(output_line[0])  # 计算bartste句的情感极性分数

        input_sent = sentiment_helper(input_line_sent)  # 根据分数判断原句的情感极性
        output_sent = sentiment_helper(output_line_sent)  # 根据分数判断bartste句的情感极性

        if input_sent == output_sent:  # 如果原句和bartste句的情感极性一致
            total_sent_count += 1

        input_sentiments_lst.append(input_sent)
        output_sentiments_lst.append(output_sent)

    final_sent_score = total_sent_count / total_lines
    # print("Final sentiment score: ", final_sent_score)

    return input_sentiments_lst, output_sentiments_lst, total_sent_count, final_sent_score


# #BLEU Scores

def calc_corpus_BLEU(references, hypotheses):
    BLEU_score = corpus_bleu(references, hypotheses)
    print("Corpus BLEU score: ", BLEU_score)
    return BLEU_score


def get_corpus_BLEU(input_path, output_path):
    references = [[(line.strip().split('\t'))[1].split()] for line in open(input_path, encoding='utf-8')]
    hypotheses = [(line.strip().split('\t'))[0].split() for line in open(output_path, encoding='utf-8')]
    corpus_BLEU_score = calc_corpus_BLEU(references, hypotheses)
    return corpus_BLEU_score


# Semantic Content Similarity Scores (CSS)


# #Perplexity & SLOR Scores


# # get_ipython().system('pip3 install flair')


def calc_perplexity(sentence):
    if len(sentence) == 1:
        # print("1sentence")
        # print(sentence)
        sentence_perplexity = language_model.calculate_perplexity(sentence + ' ')
    else:
        # print("2sentence")
        # print(sentence)
        # print("len(sentence)")  # 句或单词中字符的个数
        # print(len(sentence))
        sentence_perplexity = language_model.calculate_perplexity(sentence)
    # print(f'"{sentence}" - perplexity is {sentence_perplexity}')
    return sentence_perplexity


def calc_token_perplexities(token_lst):
    total_token_counter = 0
    token_perplexity_sum = 0
    for token in token_lst:
        if len(token) == 1:  # 如果这个单词只有一个字母
            # print("1token")
            # print(token)
            token_len = 2
            token_perplexity = calc_perplexity(token + ' ')
        else:  # 如果这个单词由多个字母组成
            # print("2token")
            # print(token)
            # print("len(token)")
            # print(len(token))
            token_len = len(token)
            token_perplexity = calc_perplexity(token)
        total_token_counter += token_len
        token_perplexity_sum += token_len * math.log(token_perplexity)
    return token_perplexity_sum, total_token_counter


# Function to calculate average actual masking rate, perplexity, and SLOR for a given file
# Note: it also calculates the individual SLOR values per generation and returns a list of them
# Note: file_type should be either "input" or "output" depending on if it contains original text or generations, respectively
def calc_input_PPL_and_SLOR(path, total_lines):
    print("Currently calculating masking rate, PPL, and SLOR for file: ", path)

    total_input_perplexity = 0
    total_input_SLOR = 0
    SLOR_lst = []

    input_lines = [line.strip().split('\t') for line in open(path, encoding='utf-8')]

    for input_line in input_lines:
        if len(input_line[1]) == 0 or len(input_line[1].split()) == 0:
            print("Error input line: ", input_line)

        else:
            input_perplexity = calc_perplexity(input_line[1])  # 原句子的PPL
            total_input_perplexity += input_perplexity

            input_tokens = input_line[1].split()
            # print("input_tokens")
            # print(input_tokens)
            input_tokens_perplexity, tokens_len = calc_token_perplexities(input_tokens)  # 用于计算SLOR
            input_SLOR = -math.log(input_perplexity) + input_tokens_perplexity / tokens_len
            SLOR_lst.append(input_SLOR)
            total_input_SLOR += input_SLOR

    avg_input_perplexity = total_input_perplexity / total_lines
    # print("Avg input perplexity: ", avg_input_perplexity)

    avg_input_SLOR = total_input_SLOR / total_lines
    # print("Avg input SLOR: ", avg_input_SLOR)

    return avg_input_perplexity, avg_input_SLOR, SLOR_lst


def calc_output_PPL_and_SLOR(path, total_lines):
    print("Currently calculating masking rate, PPL, and SLOR for file: ", path)

    total_masking = 0
    total_output_perplexity = 0
    total_output_SLOR = 0
    SLOR_lst = []

    output_lines = [line.strip().split('\t') for line in open(path, encoding='utf-8')]

    for output_line in output_lines:
        # print("output_line")
        # print(output_line)
        if len(output_line) < 3 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:
            # Below line is if condition for news headlines dataset or bartbaseline
        # if len(output_line) < 2 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:
            print("Error output line: ", output_line)

        else:
            total_masking += float(output_line[1])

            output_perplexity = calc_perplexity(output_line[0])
            total_output_perplexity += output_perplexity

            output_tokens = output_line[0].split()
            output_tokens_perplexity, tokens_len = calc_token_perplexities(output_tokens)
            output_SLOR = -math.log(output_perplexity) + output_tokens_perplexity / tokens_len
            SLOR_lst.append(output_SLOR)
            total_output_SLOR += output_SLOR

    avg_masking = total_masking / total_lines
    # print("Avg masking: ", avg_masking)

    avg_output_perplexity = total_output_perplexity / total_lines
    # print("Avg output perplexity: ", avg_output_perplexity)

    avg_output_SLOR = total_output_SLOR / total_lines
    # print("Avg output SLOR: ", avg_output_SLOR)

    return avg_masking, avg_output_perplexity, avg_output_SLOR, SLOR_lst


def get_SLOR_list(paths):
    total_SLOR_list = []
    for path in paths:
        total_SLOR_list.extend([float(line.strip()) for line in open(path, encoding='utf-8')])
    return total_SLOR_list


def fit_gaussian(lst):
    mu, std = norm.fit(lst)
    print("\nmu: ", mu, "\nstd: ", std)

    # Plot the histogram.
    plt.hist(lst, bins=25, density=True, alpha=0.6, color='g')  # 绘制直方图

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    # plt.show()

    return mu, std


def normalize_SLOR(SLOR_list, mu, std):  # 把SLOR值标准化并去除在置信区间外的SLOR值
    new_SLOR_list = []  # 在置信区间内的SLOR值
    SLOR_out = []  # 在置信区间外的SLOR值
    for score in SLOR_list:
        new_score = (score - mu) / std
        new_SLOR_list.append(new_score)
        if new_score > 3 or new_score < -3:  # 去除在置信区间外的SLOR值
            SLOR_out.append(new_score)
    return new_SLOR_list, SLOR_out


def final_normalize_SLOR(SLOR_list):  # 把这个值域为[-3, 3]的值变成值域为[0, 1]
    new_SLOR_list = []
    for score in SLOR_list:
        if score < -3:
            new_score = (-3 / 6) + 0.5
        elif score > 3:
            new_score = (3 / 6) + 0.5
        else:
            new_score = (score / 6) + 0.5
        new_SLOR_list.append(new_score)
    return new_SLOR_list


def get_avgs(final_list):
    avg = sum(final_list) / len(final_list)
    return avg


def final_SLOR_normalization(pre_SLOR_list):
    mu, std = fit_gaussian(pre_SLOR_list)  # 获得SLOR的正态分布的均值与标准差并绘制柱状图

    normalized_SLOR_list, SLOR_out = normalize_SLOR(pre_SLOR_list, mu, std)  # 把SLOR值标准化并去除在置信区间外的SLOR值
    mu_2, std_2 = fit_gaussian(normalized_SLOR_list)  # 去除在置信区间外的SLOR值后, 再求其正态分布的均值与标准差并绘制柱状图
    # print("Length of SLOR_out: ", len(SLOR_out))  # 在置信区间外的SLOR值的个数
    # print("Ratio out: ", len(SLOR_out) / len(normalized_SLOR_list))  # 在置信区间外的SLOR值占所有值的比例

    final_SLOR_list = final_normalize_SLOR(normalized_SLOR_list)  # 把这个值域为[-3, 3]的值变成值域为[0, 1]
    mu_3, std_3 = fit_gaussian(final_SLOR_list)  # 获得SLOR的正态分布的均值与标准差
    # print("Minimum: ", min(final_SLOR_list))
    # print("Maximum: ", max(final_SLOR_list), '\n')

    final_SLOR_avg = get_avgs(final_SLOR_list)
    print("final SLOR avg: ", final_SLOR_avg)


# # total_lines = 1000
# # total_lines = 2916
# original_lines = 3000
# output_line = 2993

# # input_path = "eval_reviews.txt"
# # output_path = "yelp_output_80_seq2seq.txt"

# ste_dir = r"E:\PycharmProjects\SMERTI-master"
# dataset_dir = os.path.join(ste_dir, "Amazon_Dataset")
# eval_dir = os.path.join(dataset_dir, "evaluation_data")
# method_dir = os.path.join(eval_dir, "ste_bart_mutimask")
# method_dir = os.path.join(eval_dir, "baseline_result")

# For SPA, BLEU and CSS
# output_lines_num = 3000
# input_path = os.path.join(method_dir, "reviews_filtered_input.txt")
# output_path = os.path.join(method_dir, "reviews_output_80.txt")
# For SLOR
# original_lines_num = 3000
# original_path = os.path.join(eval_dir, "eval_reviews.txt")

# SPA
# input_sent_lst, output_sent_lst, total_sent_count, final_sent_score = vader_sentiments(input_path, output_path,
#                                                                                        output_lines_num)


# BLEU
# corpus_BLEU_score = get_corpus_BLEU(input_path, output_path)


# SLOR
# _, _, pre_input_SLOR_lst = calc_input_PPL_and_SLOR(original_path, original_lines_num)  # 未标准化的原数据集的SLOR

# _, _, _, pre_output_SLOR_lst = calc_output_PPL_and_SLOR(output_path, output_lines_num)  # 未标准化的SLOR


# # final_SLOR_normalization(pre_input_SLOR_lst)
# # final_SLOR_normalization(pre_output_SLOR_lst)


def choose_dataset(dataset_name, method_name, output_lines_num):
    original_lines_num = 3000

    if dataset_name == 'News':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_headlines.txt")
        if method_name == 'baseline_result':
            method_dir = os.path.join(eval_dir, "baseline_result")
            input_path = original_path
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
        else:
            method_dir = os.path.join(eval_dir, method_name)
            input_path = os.path.join(method_dir, "news_filtered_input.txt")
        output_path1 = os.path.join(method_dir, "news_output_20.txt")
        output_path2 = os.path.join(method_dir, "news_output_40.txt")
        output_path3 = os.path.join(method_dir, "news_output_60.txt")
        output_path4 = os.path.join(method_dir, "news_output_80.txt")
    elif dataset_name == 'Amazon':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_reviews.txt")
        if method_name == 'baseline_result':
            method_dir = os.path.join(eval_dir, "baseline_result")
            input_path = original_path
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
        else:
            method_dir = os.path.join(eval_dir, method_name)
            input_path = os.path.join(method_dir, "reviews_filtered_input.txt")
        output_path1 = os.path.join(method_dir, "reviews_output_20.txt")
        output_path2 = os.path.join(method_dir, "reviews_output_40.txt")
        output_path3 = os.path.join(method_dir, "reviews_output_60.txt")
        output_path4 = os.path.join(method_dir, "reviews_output_80.txt")
    elif dataset_name == 'Yelp':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Yelp_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_reviews.txt")
        if method_name == 'baseline_result':
            method_dir = os.path.join(eval_dir, "baseline_result")
            input_path = original_path
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
        else:
            method_dir = os.path.join(eval_dir, method_name)
            input_path = os.path.join(method_dir, "reviews_filtered_input.txt")
        output_path1 = os.path.join(method_dir, "reviews_output_20.txt")
        output_path2 = os.path.join(method_dir, "reviews_output_40.txt")
        output_path3 = os.path.join(method_dir, "reviews_output_60.txt")
        output_path4 = os.path.join(method_dir, "reviews_output_80.txt")
    else:
        print("wrong dataset_name!")
        return False

    _, _, _, final_sent_score = vader_sentiments(input_path, output_path1, output_lines_num)
    print("Final sentiment 20score : ", final_sent_score)
    _, _, _, final_sent_score = vader_sentiments(input_path, output_path2, output_lines_num)
    print("Final sentiment 40score : ", final_sent_score)
    _, _, _, final_sent_score = vader_sentiments(input_path, output_path3, output_lines_num)
    print("Final sentiment 60score : ", final_sent_score)
    _, _, _, final_sent_score = vader_sentiments(input_path, output_path4, output_lines_num)
    print("Final sentiment 80score : ", final_sent_score)

    avg_input_perplexity, avg_input_SLOR, _ = calc_input_PPL_and_SLOR(original_path,
                                                                      original_lines_num)  # 未标准化的原数据集的SLOR
    print("input:  Avg input perplexity: ", avg_input_perplexity, "    Avg input SLOR: ", avg_input_SLOR)
    avg_masking, avg_output_perplexity, avg_output_SLOR, _ = calc_output_PPL_and_SLOR(output_path1,
                                                                                      output_lines_num)
    print("20:  Avg masking: ", avg_masking, "   Avg output perplexity: ", avg_output_perplexity,
          "  Avg output SLOR: ", avg_output_SLOR)
    avg_masking, avg_output_perplexity, avg_output_SLOR, _ = calc_output_PPL_and_SLOR(output_path2,
                                                                                      output_lines_num)
    print("40:  Avg masking: ", avg_masking, "   Avg output perplexity: ", avg_output_perplexity,
          "  Avg output SLOR: ", avg_output_SLOR)
    avg_masking, avg_output_perplexity, avg_output_SLOR, _ = calc_output_PPL_and_SLOR(output_path3,
                                                                                      output_lines_num)
    print("60:  Avg masking: ", avg_masking, "   Avg output perplexity: ", avg_output_perplexity,
          "  Avg output SLOR: ", avg_output_SLOR)
    avg_masking, avg_output_perplexity, avg_output_SLOR, _ = calc_output_PPL_and_SLOR(output_path4,
                                                                                      output_lines_num)
    print("80:  Avg masking: ", avg_masking, "   Avg output perplexity: ", avg_output_perplexity,
          "  Avg output SLOR: ", avg_output_SLOR)


# Dataset_name = 'Amazon'
Dataset_name = 'Yelp'
# Method_name = 'ste-bart-mutimask-coordinate-similiar'
# Method_name = 'ste-bart-mutimask-hyper-similiar'
# Method_name = 'ste-bart-mutimask-hypo-similiar'
# Method_name = 'ste-bart-large-mutimask-coordinate-similiar'
# Method_name = 'ste-bart-large-coordinate-similiar'
# Method_name = 'ste-bart-large-hyper-similiar'
# Method_name = 'ste-bart-large-hypo-similiar'
# Method_name = 'baseline-bart'
Method_name = 'baseline_result'
# Method_name = 'baseline-bart-large'

Output_lines_num = 3000
# Output_lines_num = 2964
# Output_lines_num = 2998
# Output_lines_num = 2993
choose_dataset(Dataset_name, Method_name, Output_lines_num)


# test_path = r'E:\PycharmProjects\SMERTI-master\News_Dataset\test.txt'
# avg_input_perplexity, avg_input_SLOR, _ = calc_input_PPL_and_SLOR(test_path,
#                                                                   1400)  # 未标准化的原数据集的SLOR
# print("input:  Avg input perplexity: ", avg_input_perplexity, "    Avg input SLOR: ", avg_input_SLOR)
