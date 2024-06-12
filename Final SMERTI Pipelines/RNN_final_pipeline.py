#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('nvidia-smi')

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# ##USE Entity Similarity

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


get_ipython().system('pip3 install stanfordcorenlp')
from string import punctuation
from collections import OrderedDict

import nltk
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')


def parse_entity(entity):
  labels_list = []
  for subtree in entity.subtrees():
    if subtree.label() != 'ROOT':
      label = subtree.label()
      if len(label) >= 3:
        labels_list.append(label[:2])
      else:
        labels_list.append(label)
  return labels_list


def extract_leaves(tree, labels_list):
    leaves_list = []
    for i in tree.subtrees():
      if i.label()[:2] in labels_list:
        leaves_list.append(i.leaves())
    return leaves_list


def create_entities_list(list_of_lists):
    entities_list = []
    for entity_list in list_of_lists:
        entity = ' '.join(entity_list).replace(" '", "'")
        if entity not in entities_list:
          entities_list.append(entity)
    return entities_list


# In[ ]:


def get_similarity_matrix(embedded_text, text_input, input_list):
  message_embeddings = session.run(
      embedded_text, feed_dict={text_input: input_list})
  corr = np.inner(message_embeddings, message_embeddings)
  return corr


# In[ ]:


def process_text(s):
    s = s.lower().strip()
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)    
    s = re.sub("[^a-zA-Z.!?,'']+", ' ', s)
    for p in punctuation:
      if p not in ["'", "[", "]"]:
        s = s.replace(p, " " + p + " ")       
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s

    
def sentence_to_words(sentence):
  word_list = sentence.split()
  return word_list


# In[ ]:


def process_sentence(initial_sentence):
  initial_sentence = process_text(initial_sentence)
  return initial_sentence


def process_entity(replacement_entity):
  replacement_entity = process_text(replacement_entity)
  return replacement_entity


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


def get_trees(initial_sentence, replacement_entity):
  entity = Tree.fromstring(nlp.parse(replacement_entity))
  labels_list = parse_entity(entity)
  
  full_leaves_list = []
  sentences = re.split('[?.!]', initial_sentence)
  sentences = list(filter(lambda x: x not in ['',' '], sentences))
  for sentence in sentences:
    sentence = Tree.fromstring(nlp.parse(sentence))
    leaves = extract_leaves(sentence, labels_list)
    full_leaves_list = full_leaves_list + leaves
  return full_leaves_list

  
def create_input_list(entities_list, word_list, replacement_entity):
  entities_list = list(filter(lambda x: x not in word_list, entities_list))
  full_input_list = word_list + entities_list + [replacement_entity]
  full_input_list = list(filter(lambda x: x not in punctuation, full_input_list))
  full_input_list = list(OrderedDict.fromkeys(full_input_list))
  return full_input_list


# In[ ]:


def get_replacement_similarity(similarity_matrix):
  replacement_entity_similarity = similarity_matrix[len(similarity_matrix)-1]
  return replacement_entity_similarity


def get_index_max(replacement_entity_similarity):
  index_max = np.argmax(replacement_entity_similarity[:-1])
  return index_max


def get_full_index_max(full_input_list, replaced_entity):
  full_index_max = full_input_list.index(replaced_entity)
  return full_index_max
  

def get_original_similarity(similarity_matrix, index_max):
  original_similarity = similarity_matrix[index_max]
  return original_similarity


def generate_new_sentence(entities_list, replacement_entity, initial_sentence, index_max):
  replaced_entity = entities_list[index_max]
  new_sentence = re.sub(r"\b%s\b" % replaced_entity, replacement_entity, initial_sentence)
  return replaced_entity, new_sentence


def get_new_length(new_sentence):
  new_word_list = sentence_to_words(new_sentence)
  new_list_len = len(list(filter(lambda x: x not in punctuation, new_word_list)))
  return new_list_len


# In[ ]:


def get_masked_sentence(mask_threshold, similarity_threshold, length, similarity_vector_old, word_list, 
                      index_max, original_len, similar_words, new_sentence, mask_sentence, replacement_entity):
  
  if length <= int(round((original_len * mask_threshold))):
    return mask_sentence, length/original_len
  
  else:
    counter = 0
    indices_list = []

    for score in similarity_vector_old:
      if score > similarity_threshold and counter != index_max and counter not in indices_list:
        indices_list.append(counter)
      counter += 1
    
    similar_words = []
    for index in indices_list:
        similar_words.append(word_list[index])
    similar_words = list(filter(lambda x: x not in punctuation, similar_words))
    similar_words = list(set(similar_words))
    similar_words.sort(key=lambda x: len(x.split()), reverse=True)
    
    temp_mask_num, temp_mask_sentence = mask_similar_words(similar_words, new_sentence, replacement_entity)
    
    return get_masked_sentence(mask_threshold, similarity_threshold + 0.05, temp_mask_num, similarity_vector_old, 
                      word_list, index_max, original_len, similar_words, new_sentence, temp_mask_sentence, replacement_entity)
  

def mask_similar_words(similar_words, sentence, replacement_entity):
  sentence_temp = sentence
  masked_sentence = ""
  mask_counter = 0
  word_counter = 0
  if len(similar_words) == 0:
    masked_sentence = sentence_temp
  else:
    for word in similar_words:
      if word not in replacement_entity and replacement_entity not in word:
        sentence_temp = re.sub(r"\b%s\b" % word, "[mask]", sentence_temp)
        masked_sentence = sentence_temp
        temp_mask_counter = masked_sentence.count("[mask]")
        if temp_mask_counter > mask_counter:
          mask_counter += 1
          num_of_words = len(word.split())
          word_counter += num_of_words
  return word_counter, masked_sentence


def mask_groupings(masked_list):
  masked_group_list = []
  previous_element = ""
  for element in masked_list:
    if element != "[mask]":
      masked_group_list.append(element)
    elif element == "[mask]":
      if element != previous_element:
        masked_group_list.append(element)
    previous_element = element
  return masked_group_list
    

def mask_fnc(replacement_entity_similarity, original_similarity, full_input_list, index_max, original_len, new_sentence, mask_threshold, similarity_threshold, replacement_entity):
  masked_sentence, final_mask_rate = get_masked_sentence(mask_threshold, similarity_threshold, 1000, original_similarity[:-1], full_input_list, index_max, original_len, [], new_sentence, "", replacement_entity)
  masked_word_list = masked_sentence.split()
  masked_group_list = mask_groupings(masked_word_list)
  masked_group_sentence = ' '.join(masked_group_list)
  return masked_group_sentence, final_mask_rate


# In[ ]:


def main_USE_function(input_sentence, replacement_entity, embedded_text, text_input):
    initial_sentence = process_sentence(input_sentence)
    replacement_entity = process_entity(replacement_entity)
    word_list = get_word_list(initial_sentence)
    original_len = get_original_len(word_list)
    
    leaves = get_trees(initial_sentence, replacement_entity)
    entities_list = create_entities_list(leaves)
    
    if entities_list == []:
      entities_list = word_list
      
    full_entities_list = entities_list + [replacement_entity]
    entity_similarities = get_similarity_matrix(embedded_text, text_input, full_entities_list)
    replacement_similarities = get_replacement_similarity(entity_similarities)
    index_max = get_index_max(replacement_similarities)
    replaced_entity, new_sentence = generate_new_sentence(entities_list, replacement_entity, initial_sentence, index_max)
    
    full_input_list = create_input_list(entities_list, word_list, replacement_entity)
    full_index_max = get_full_index_max(full_input_list, replaced_entity)
    similarity_matrix = get_similarity_matrix(embedded_text, text_input, full_input_list)
    full_replacement_similarities = get_replacement_similarity(similarity_matrix)
    original_similarity = get_original_similarity(similarity_matrix, full_index_max)
    
    masked_group_sentence_1, final_mask_rate_1 = mask_fnc(full_replacement_similarities, original_similarity, full_input_list, 
                                                  full_index_max, original_len, new_sentence, 0.2, 0.4, replacement_entity)
    masked_group_sentence_2, final_mask_rate_2 = mask_fnc(full_replacement_similarities, original_similarity, full_input_list, 
                                                  full_index_max, original_len, new_sentence, 0.4, 0.3, replacement_entity)
    masked_group_sentence_3, final_mask_rate_3 = mask_fnc(full_replacement_similarities, original_similarity, full_input_list, 
                                                  full_index_max, original_len, new_sentence, 0.6, 0.2, replacement_entity)
    masked_group_sentence_4, final_mask_rate_4 = mask_fnc(full_replacement_similarities, original_similarity, full_input_list, 
                                                  full_index_max, original_len, new_sentence, 0.8, 0.1, replacement_entity)
    masked_group_sentences = [masked_group_sentence_1, masked_group_sentence_2, masked_group_sentence_3, masked_group_sentence_4]
    final_mask_rates = [final_mask_rate_1, final_mask_rate_2, final_mask_rate_3, final_mask_rate_4]
    
    return masked_group_sentences, final_mask_rates


# ##RNN for Mask Filling

# Based on code from: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

# In[ ]:


#import necessities

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import ast


#use CUDA (GPU) if available, else CPU

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
  print("Using CUDA from GPU")
  
get_ipython().system('nvidia-smi')


# In[ ]:


main_path = "/content/drive/My Drive/"
corpus_name = "News_Dataset"
corpus = os.path.join(main_path, corpus_name)
save_dir = os.path.join(corpus, "output")
data_dir = os.path.join(corpus, "data/seq2seq_data")
formatted_file = os.path.join(data_dir, "masked_train_headlines.txt")
formatted_file_valid = os.path.join(data_dir, "masked_val_headlines.txt")

MAX_LENGTH = 100


# In[ ]:


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# In[ ]:


from string import punctuation

# Turn a Unicode string to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)    
    s = re.sub("[^a-zA-Z.!?,\[\]'']+", ' ', s)
    for p in punctuation:
      if p not in ["'", "[", "]"]:
        s = s.replace(p, " " + p + " ")       
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s

  
def readVocs(formatted_file, corpus_name):
    print("Reading lines...")
    lines = open(formatted_file, encoding='utf-8').        read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

  
def getValidPairs(formatted_file_valid):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(formatted_file_valid, encoding='utf-8').        read().strip().split('\n')
    valid_pairs = [l.split('\t') for l in lines]
    return valid_pairs  
  
  
# Returns True only if both sentences in pair p are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

  
# Filter pairs using filterPair function above
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

  
# Using the functions above, return a populated Voc object and pairs list
def loadPrepareData(corpus, corpus_name, formatted_file, formatted_file_valid, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(formatted_file, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    
    valid_pairs = getValidPairs(formatted_file_valid)
    valid_pairs = filterPairs(valid_pairs)
    for valid_pair in valid_pairs:
        voc.addSentence(valid_pair[0])
        voc.addSentence(valid_pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs, valid_pairs

  
def indexesFromSentence(voc, sentence):
  voc_list = []
  for word in sentence.split(' '):
    try:
      voc_index = voc.word2index[word]
      voc_list.append(voc_index)
    except KeyError:
      continue
  return voc_list + [EOS_token]
  
  
# Load and assemble voc and pairs
voc, pairs, valid_pairs = loadPrepareData(corpus, corpus_name, formatted_file, formatted_file_valid, save_dir)
# Print out some pairs
print("\nPairs:")
for pair in pairs[:10]:
    print(pair)

print("\nValidation Pairs:") 
for valid_pair in valid_pairs[:10]:
    print(valid_pair)


# In[ ]:


MIN_COUNT = 0    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in both their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# In[ ]:


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# In[ ]:


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# In[ ]:


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# In[ ]:


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc, input_sentence):
    # Normalize sentence
    input_sentence = normalizeString(input_sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words_trimmed = []
    stop = 0
    for word in output_words:
      if word == 'EOS' or word == 'PAD':
        stop = 1
      elif stop == 0:
        output_words_trimmed.append(word)
    final_sentence = ' '.join(output_words_trimmed)
    print('Output:', ' '.join(output_words_trimmed))
    return final_sentence


# In[ ]:


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
#loadFilename = None
checkpoint_iter = 20000 # Checkpoint that you wish to load from
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# In[ ]:


# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)


# ##Example

# In[ ]:


#Example (for news headlines)

user_input = "Obama is the president of America"
replacement_entity = "Trump"

masked_sentences, mask_rates = main_USE_function(user_input, replacement_entity, embedded_text, text_input)
print("\nFinal masked sentences: ", masked_sentences)
print("Final mask rates: ", mask_rates, '\n')

output_example = evaluate_input(encoder, decoder, searcher, voc, masked_sentences[0])


# #Evaluation Data Preparation

# ## Write Pipeline Outputs for Evaluation Lines

# In[ ]:


eval_dir = os.path.join(corpus, "data/evaluation_data/")
eval_path = os.path.join(eval_dir, "eval_headlines.txt")

path_20 = os.path.join(eval_dir, "news_output_20_seq2seq.txt")
path_40 = os.path.join(eval_dir, "news_output_40_seq2seq.txt")
path_60 = os.path.join(eval_dir, "news_output_60_seq2seq.txt")
path_80 = os.path.join(eval_dir, "news_output_80_seq2seq.txt")


# In[ ]:


def get_eval_lines(eval_path):
    print("Reading lines...")
    lines = open(eval_path, encoding='utf-8').        read().strip().split('\n')
    eval_lines = [l.split('\t') for l in lines]
    return eval_lines
  
  
eval_lines = get_eval_lines(eval_path)

for eval_line in eval_lines[:10]:
    print(eval_line)


# In[ ]:


def write_results(eval_lines, path_20, path_40, path_60, path_80, counter_start):
  f_20 = open(path_20, 'a')
  f_40 = open(path_40, 'a')
  f_60 = open(path_60, 'a')
  f_80 = open(path_80, 'a')
  
  counter = counter_start
  
  for line in eval_lines:
    print("\nCurrently evaluating line {}".format(counter))
    print("Line: ", line)
      
    replacement_entity = line[0]
    user_input = line[1]
    masked_sentences, mask_rates = main_USE_function(user_input, replacement_entity, embedded_text, text_input)
    
    output_20 = evaluate_input(encoder, decoder, searcher, voc, masked_sentences[0])
    output_40 = evaluate_input(encoder, decoder, searcher, voc, masked_sentences[1])
    output_60 = evaluate_input(encoder, decoder, searcher, voc, masked_sentences[2])
    output_80 = evaluate_input(encoder, decoder, searcher, voc, masked_sentences[3])
    
    mask_rate_20 = mask_rates[0]
    mask_rate_40 = mask_rates[1]
    mask_rate_60 = mask_rates[2]
    mask_rate_80 = mask_rates[3]
    
    ##For news headlines dataset:
    f_20.write(output_20 + '\t' + str(round(mask_rate_20, 3)) + '\n')
    f_40.write(output_40 + '\t' + str(round(mask_rate_40, 3)) + '\n')
    f_60.write(output_60 + '\t' + str(round(mask_rate_60, 3)) + '\n')
    f_80.write(output_80 + '\t' + str(round(mask_rate_80, 3)) + '\n')
    
    ##For reviews:
    #f_20.write(output_20 + '\t' + str(round(mask_rate_20, 3)) + '\t' + line[2] + '\n')
    #f_40.write(output_40 + '\t' + str(round(mask_rate_40, 3)) + '\t' + line[2] + '\n')
    #f_60.write(output_60 + '\t' + str(round(mask_rate_60, 3)) + '\t' + line[2] + '\n')
    #f_80.write(output_80 + '\t' + str(round(mask_rate_80, 3)) + '\t' + line[2] + '\n')
    
    counter += 1
  
  f_20.close()
  f_40.close()
  f_60.close()
  f_80.close()
  

write_results(eval_lines, path_20, path_40, path_60, path_80, 1)

