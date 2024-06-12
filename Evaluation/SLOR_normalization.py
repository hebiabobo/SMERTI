#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('nvidia-smi')

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


# In[ ]:


def get_SLOR_list(paths):
  total_SLOR_list = []
  for path in paths:
    total_SLOR_list.extend([float(line.strip()) for line in open(path)])
  return total_SLOR_list


# In[ ]:


def fit_gaussian(lst):
  mu, std = norm.fit(lst)
  print("\nmu: ", mu, "\nstd: ", std)

  # Plot the histogram.
  plt.hist(lst, bins=25, density=True, alpha=0.6, color='g')

  # Plot the PDF.
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2)
  title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
  plt.title(title)

  plt.show()
  
  return mu, std


# In[ ]:


def normalize_SLOR(SLOR_list, mu, std):
  new_SLOR_list = []
  SLOR_out = []
  for score in SLOR_list:
    new_score = (score - mu) / std
    new_SLOR_list.append(new_score)
    if new_score > 3 or new_score < -3:
      SLOR_out.append(new_score)
  return new_SLOR_list, SLOR_out


# In[ ]:


def final_normalize_SLOR(SLOR_list):
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


# In[ ]:


def chunks(l, n):
  n = max(1, n)
  return [l[i:i+n] for i in range(0, len(l), n)]


def get_chunk_avgs(chunks):
  chunk_avg_list = []
  for chunk in chunks:
    chunk_avg = sum(chunk) / len(chunk)
    print("Chunk avg: ", chunk_avg)
    chunk_avg_list.append(chunk_avg)
  return chunk_avg_list


# In[ ]:


path1 = "SLOR_list_amazon.txt"
path2 = "SLOR_list_yelp.txt"
path3 = "SLOR_list_news.txt"

total_SLOR_list = get_SLOR_list([path1, path2, path3])
mu, std = fit_gaussian(total_SLOR_list)

normalized_SLOR_list, SLOR_out = normalize_SLOR(total_SLOR_list, mu, std)
mu_2, std_2 = fit_gaussian(normalized_SLOR_list)
print("Length of SLOR_out: ", len(SLOR_out))
print("Ratio out: ", len(SLOR_out) / len(normalized_SLOR_list))

final_SLOR_list = final_normalize_SLOR(normalized_SLOR_list)
mu_3, std_3 = fit_gaussian(final_SLOR_list)
print("Minimum: ", min(final_SLOR_list))
print("Maximum: ", max(final_SLOR_list), '\n')

SLOR_chunks = chunks(final_SLOR_list, 1000)
chunk_avgs = get_chunk_avgs(SLOR_chunks)

