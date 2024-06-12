#!/usr/bin/env python
# coding: utf-8

# Note: format of the files inputted to these functions as follows:
# 
# -Input_path: each line, when split by tab, contains an RE and original piece of text (either review or headline), one per line
# 
# -Output_path: each line, when split by tab, contains a generated piece of text and the actual masking rate for that generation as the first two elements, one per line

# In[ ]:


get_ipython().system('nvidia-smi')

from google.colab import drive
drive.mount('/content/drive')


# #Sentiment Scores

# In[ ]:


import nltk

get_ipython().system('pip3 install vaderSentiment')
nltk.download('vader_lexicon')


# In[ ]:


import pandas as pd 
import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
  score = analyser.polarity_scores(sentence)
  #print("{:-<40} {}".format(sentence, str(score)))
  return score


def sentiment_helper(line_sent):
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
  input_lines = [line.strip().split('\t') for line in open(input_path)]
  output_lines = [line.strip().split('\t') for line in open(output_path)]
  
  for input_line, output_line in zip(input_lines, output_lines):
    input_line_sent = sentiment_analyzer_scores(input_line[1])
    output_line_sent = sentiment_analyzer_scores(output_line[0])
    
    input_sent = sentiment_helper(input_line_sent)
    output_sent = sentiment_helper(output_line_sent)

    if input_sent == output_sent:
      total_sent_count += 1
      
    input_sentiments_lst.append(input_sent)
    output_sentiments_lst.append(output_sent)
  
  final_sent_score = total_sent_count / total_lines
  print("Final sentiment score: ", final_sent_score)
  
  return input_sentiments_lst, output_sentiments_lst, total_sent_count, final_sent_score


total_lines = 1000

input_path = "eval_reviews.txt"
output_path = "yelp_output_80_seq2seq.txt"

input_sent_lst, output_sent_lst, total_sent_count, final_sent_score = vader_sentiments(input_path, output_path, total_lines)


# #BLEU Scores

# In[ ]:


import nltk
from nltk.translate.bleu_score import corpus_bleu


def calc_corpus_BLEU(references, hypotheses):
  BLEU_score = corpus_bleu(references, hypotheses)
  print("Corpus BLEU score: ", BLEU_score)
  return BLEU_score


def get_corpus_BLEU(input_path, output_path):
  references = [[(line.strip().split('\t'))[1].split()] for line in open(input_path)]
  hypotheses = [(line.strip().split('\t'))[0].split() for line in open(output_path)]
  corpus_BLEU_score = calc_corpus_BLEU(references, hypotheses)
  return corpus_BLEU_score


# In[ ]:


input_path = "eval_reviews.txt"
output_path = "yelp_output_20_seq2seq.txt"

corpus_BLEU_score = get_corpus_BLEU(input_path, output_path)


# #Semantic Content Similarity Scores (CSS)

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns

tf.logging.set_verbosity(tf.logging.WARN)


# In[ ]:


g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
  g.finalize()

session = tf.Session(graph=g)
session.run(init_op)


# In[ ]:


def get_similarity_score(embedded_text, text_input, input_list):
  message_embeddings = session.run(
      embedded_text, feed_dict={text_input: input_list})
  corr = np.inner(message_embeddings, message_embeddings)
  sim_score = corr[0][1]
  #print("Similarity score for {}: ".format(input_list), sim_score, '\n')
  return sim_score


# In[ ]:


#Get average CSS of the original text
def get_input_CSS(input_path, embedded_text, text_input):
  print("\n\nCurrently calculating scores for file: ", input_path)
  counter = 1
  scores_lst = []
  input_lines = [line.strip().split('\t') for line in open(input_path)]
  for line in input_lines:
    #print("Evaluating line: ", counter)
    sim_score = get_similarity_score(embedded_text, text_input, [line[0], line[1]])
    scores_lst.append(sim_score)
    counter += 1
  avg_score = np.mean(scores_lst)
  print("Average score: ", avg_score)
  return avg_score


# In[ ]:


#Get average CSS of generated text by model(s)
def get_output_CSS(input_path, output_path, embedded_text, text_input):
  print("\n\nCurrently calculating scores for file: ", output_path)
  counter = 1
  scores_lst = []
  input_lines = [line.strip().split('\t') for line in open(input_path)]
  output_lines = [line.strip().split('\t') for line in open(output_path)]
  for input_line, output_line in zip(input_lines, output_lines):
    #print("Evaluating line: ", counter)
    sim_score = get_similarity_score(embedded_text, text_input, [input_line[0], output_line[0]])
    scores_lst.append(sim_score)
    counter += 1
  avg_score = np.mean(scores_lst)
  print("Average score: ", avg_score)
  return avg_score


# In[ ]:


input_path = "eval_reviews.txt"
output_path = "yelp_output_20_seq2seq.txt"

input_CSS = get_input_CSS(input_path, embedded_text, text_input)
output_CSS = get_output_CSS(input_path, output_path, embedded_text, text_input)


# #Perplexity & SLOR Scores

# In[ ]:


get_ipython().system('pip3 install flair')


# In[ ]:


from flair.embeddings import FlairEmbeddings
import math
import io

# get language model
language_model = FlairEmbeddings('news-forward').lm


def calc_perplexity(sentence):
  if len(sentence) == 1:
    sentence_perplexity = language_model.calculate_perplexity(sentence + ' ')
  else:
    sentence_perplexity = language_model.calculate_perplexity(sentence)
  #print(f'"{sentence}" - perplexity is {sentence_perplexity}')
  return sentence_perplexity


def calc_token_perplexities(token_lst):
  total_token_counter = 0
  token_perplexity_sum = 0
  for token in token_lst:
    if len(token) == 1:
      token_len = 2 
      token_perplexity = calc_perplexity(token + ' ')
    else:
      token_len = len(token)
      token_perplexity = calc_perplexity(token)
    total_token_counter += token_len
    token_perplexity_sum += token_len*math.log(token_perplexity)
  return token_perplexity_sum, total_token_counter


# In[ ]:


#Function to calculate average actual masking rate, perplexity, and SLOR for a given file
#Note: it also calculates the individual SLOR values per generation and returns a list of them
#Note: file_type should be either "input" or "output" depending on if it contains original text or generations, respectively
def calc_metrics(path, total_lines, file_type):
  print("Currently calculating masking rate, PPL, and SLOR for file: ", path)

  total_masking = 0
  avg_masking = 0
  total_input_perplexity = 0
  avg_input_perplexity = 0
  total_input_SLOR = 0
  avg_input_SLOR = 0
  total_output_perplexity = 0
  avg_output_perplexity = 0
  total_output_SLOR = 0
  avg_output_SLOR = 0
  SLOR_lst = []
  
  if file_type == "input":
    input_lines = [line.strip().split('\t') for line in open(path)]
  
    for input_line in input_lines:
      if len(input_line[1]) == 0 or len(input_line[1].split()) == 0:
        print("Error input line: ", input_line)
      
      else:
        input_perplexity = calc_perplexity(input_line[1])
        total_input_perplexity += input_perplexity

        input_tokens = input_line[1].split()
        input_tokens_perplexity, tokens_len = calc_token_perplexities(input_tokens)
        input_SLOR = -math.log(input_perplexity) + input_tokens_perplexity / tokens_len
        SLOR_lst.append(input_SLOR)
        total_input_SLOR += input_SLOR

    avg_input_perplexity = total_input_perplexity / total_lines
    print("Avg input perplexity: ", avg_input_perplexity)

    avg_input_SLOR = total_input_SLOR / total_lines
    print("Avg input SLOR: ", avg_input_SLOR)

  else:
    output_lines = [line.strip().split('\t') for line in open(path)]
      
    for output_line in output_lines:
      if len(output_line) < 3 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:
      #Below line is if condition for news headlines dataset
      #if len(output_line) < 2 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:
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
    print("Avg masking: ", avg_masking)

    avg_output_perplexity = total_output_perplexity / total_lines
    print("Avg output perplexity: ", avg_output_perplexity)

    avg_output_SLOR = total_output_SLOR / total_lines
    print("Avg output SLOR: ", avg_output_SLOR)

  return avg_masking, avg_input_perplexity, avg_input_SLOR, avg_output_perplexity, avg_output_SLOR, SLOR_lst


# In[ ]:


#Writes the values in lst to a file (can be used to write SLOR_lst to a file)
def write_output_lst(lst, path):
  f = io.open(path, "w", encoding = 'utf-8')
  print("Currently writing lines to file ...")
  f.write('\n'.join(str(round(x, 10)) for x in lst))
  f.close()
  print("Lines successfully written to file!")

