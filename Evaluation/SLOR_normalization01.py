#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


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


def chunks(l, n):  # 在list l中每n个分一组(一个list)
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_chunk_avgs(chunks):
    chunk_avg_list = []
    for chunk in chunks:
        chunk_avg = sum(chunk) / len(chunk)
        print("Chunk avg: ", chunk_avg)
        chunk_avg_list.append(chunk_avg)
    return chunk_avg_list


def get_avgs(final_list):
    avg = sum(final_list) / len(final_list)
    return avg


# In[ ]:


# path1 = "SLOR_list_amazon.txt"
# path2 = "SLOR_list_yelp.txt"
# path3 = "SLOR_list_news.txt"

path1 = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data\SLOR_list_amazon.txt"
path2 = r"E:\PycharmProjects\SMERTI-master\Yelp_Dataset\evaluation_data\SLOR_list_yelp.txt"
path3 = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data\SLOR_list_news.txt"

path4 = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data\SLOR_list_input.txt"

# total_SLOR_list = get_SLOR_list([path1, path2, path3])
total_SLOR_list = get_SLOR_list([path4])
print("total_SLOR_list")
print(total_SLOR_list)
mu, std = fit_gaussian(total_SLOR_list)  # 获得SLOR的正态分布的均值与标准差并绘制柱状图

normalized_SLOR_list, SLOR_out = normalize_SLOR(total_SLOR_list, mu, std)  # 把SLOR值标准化并去除在置信区间外的SLOR值
mu_2, std_2 = fit_gaussian(normalized_SLOR_list)  # 去除在置信区间外的SLOR值后, 再求其正态分布的均值与标准差并绘制柱状图
print("Length of SLOR_out: ", len(SLOR_out))  # 在置信区间外的SLOR值的个数
print("Ratio out: ", len(SLOR_out) / len(normalized_SLOR_list))  # 在置信区间外的SLOR值占所有值的比例

final_SLOR_list = final_normalize_SLOR(normalized_SLOR_list)  # 把这个值域为[-3, 3]的值变成值域为[0, 1]
mu_3, std_3 = fit_gaussian(final_SLOR_list)  # 获得SLOR的正态分布的均值与标准差
print("Minimum: ", min(final_SLOR_list))
print("Maximum: ", max(final_SLOR_list), '\n')

final_SLOR_avg = get_avgs(final_SLOR_list)
print("final SLOR avg: ", final_SLOR_avg)

SLOR_chunks = chunks(final_SLOR_list, 1000)  # 每1000个分一组
print("SLOR_chunks")
print(SLOR_chunks)
chunk_avgs = get_chunk_avgs(SLOR_chunks)  # 每组计算SLOR平均值
