#!/usr/bin/env python
# coding: utf-8


from tensorflow.python.client import device_lib

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
    embed = hub.Module(r"E:\tensorflow_hub_objects\universal-sentence-encoder-large_3")
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

session = tf.Session(graph=g)
session.run(init_op)


from string import punctuation
from collections import OrderedDict

import nltk
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP

# nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
nlp = StanfordCoreNLP(r'E:\stanfordcorenlp\stanford-corenlp-full-2018-02-27')


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
    sentences = list(filter(lambda x: x not in ['', ' '], sentences))
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
    replacement_entity_similarity = similarity_matrix[len(similarity_matrix) - 1]
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
        return mask_sentence, length / original_len

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
                                   word_list, index_max, original_len, similar_words, new_sentence, temp_mask_sentence,
                                   replacement_entity)


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
                sentence_temp = re.sub(r"\b%s\b" % word, "<mask>", sentence_temp)
                masked_sentence = sentence_temp
                temp_mask_counter = masked_sentence.count("<mask>")
                if temp_mask_counter > mask_counter:
                    mask_counter += 1
                    num_of_words = len(word.split())
                    word_counter += num_of_words
    return word_counter, masked_sentence


def mask_groupings(masked_list):
    masked_group_list = []
    previous_element = ""
    for element in masked_list:
        if element != "<mask>":
            masked_group_list.append(element)
        elif element == "<mask>":
            if element != previous_element:
                masked_group_list.append(element)
        previous_element = element
    return masked_group_list


def mask_fnc(replacement_entity_similarity, original_similarity, full_input_list, index_max, original_len, new_sentence,
             mask_threshold, similarity_threshold, replacement_entity):
    masked_sentence, final_mask_rate = get_masked_sentence(mask_threshold, similarity_threshold, 1000,
                                                           original_similarity[:-1], full_input_list, index_max,
                                                           original_len, [], new_sentence, "", replacement_entity)
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
    replaced_entity, new_sentence = generate_new_sentence(entities_list, replacement_entity, initial_sentence,
                                                          index_max)

    full_input_list = create_input_list(entities_list, word_list, replacement_entity)
    full_index_max = get_full_index_max(full_input_list, replaced_entity)
    similarity_matrix = get_similarity_matrix(embedded_text, text_input, full_input_list)
    full_replacement_similarities = get_replacement_similarity(similarity_matrix)
    original_similarity = get_original_similarity(similarity_matrix, full_index_max)

    masked_group_sentence_1, final_mask_rate_1 = mask_fnc(full_replacement_similarities, original_similarity,
                                                          full_input_list,
                                                          full_index_max, original_len, new_sentence, 0.2, 0.4,
                                                          replacement_entity)
    masked_group_sentence_2, final_mask_rate_2 = mask_fnc(full_replacement_similarities, original_similarity,
                                                          full_input_list,
                                                          full_index_max, original_len, new_sentence, 0.4, 0.3,
                                                          replacement_entity)
    masked_group_sentence_3, final_mask_rate_3 = mask_fnc(full_replacement_similarities, original_similarity,
                                                          full_input_list,
                                                          full_index_max, original_len, new_sentence, 0.6, 0.2,
                                                          replacement_entity)
    masked_group_sentence_4, final_mask_rate_4 = mask_fnc(full_replacement_similarities, original_similarity,
                                                          full_input_list,
                                                          full_index_max, original_len, new_sentence, 0.8, 0.1,
                                                          replacement_entity)
    masked_group_sentences = [masked_group_sentence_1, masked_group_sentence_2, masked_group_sentence_3,
                              masked_group_sentence_4]
    final_mask_rates = [final_mask_rate_1, final_mask_rate_2, final_mask_rate_3, final_mask_rate_4]

    return masked_group_sentences, final_mask_rates


# ##Transformer for Mask Filling

# Based on code from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")
# get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:


cuda = torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")
device = torch.device("cpu")
# if cuda:
#     print("Using CUDA from GPU")


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[ ]:


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# In[ ]:


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[ ]:


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# In[ ]:


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[ ]:


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# In[ ]:


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# In[ ]:


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# In[ ]:


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[ ]:


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# In[ ]:


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# In[ ]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# In[ ]:


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# In[ ]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# In[ ]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


# In[ ]:


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    # model = model.cuda()
    return model


# In[ ]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src_mask = src_mask.to(device)
    model = model.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# In[ ]:


# For data loading.
import os
from torchtext import data, datasets
from torchtext.data import Field, BucketIterator, TabularDataset

main_path = r"E:\PycharmProjects\SMERTI-master"

corpus_name = "Yelp_Dataset"
corpus = os.path.join(main_path, corpus_name)
save_dir = os.path.join(corpus, r"output\transformer")
data_path = os.path.join(corpus, r"data\transformer_data")
os.chdir(data_path)

# In[ ]:


from string import punctuation
import re


def process_text_user(s):
    s = s.lower().strip()
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)
    s = re.sub("[^a-zA-Z.!?,\[\]'']+", ' ', s)
    for p in punctuation:
        if p not in ["'", "[", "]"]:
            s = s.replace(p, " " + p + " ")
            # s = re.sub('([.!?,])', ' \1', s)
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(pad_token=BLANK_WORD)
TGT = data.Field(init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

# In[ ]:


data_fields = [('src', SRC), ('trg', TGT)]
train, val, test = data.TabularDataset.splits(path=data_path, train='train.csv', validation='val.csv', test='test.csv',
                                              format='csv', fields=data_fields)

SRC.build_vocab(train, val)
TGT.build_vocab(train, val)



# ## Example

# In[ ]:


# Example (for news headlines)

# user_input = "Obama is president of America"
# replacement_entity = "Trump"
#
# masked_sentences, mask_rates = main_USE_function(user_input, replacement_entity, embedded_text, text_input)
# print("\nFinal masked sentences: ", masked_sentences)
# print("Final mask rates: ", mask_rates, '\n')
#
# output_example = evaluate_input(masked_sentences[0])

# #Evaluation Data Preparation
#

# ## Write Pipeline Outputs for Evaluation Lines

# In[ ]:


eval_dir = os.path.join(corpus, "evaluation_data")
# eval_path = os.path.join(eval_dir, "eval_headlines.txt")
eval_path = os.path.join(eval_dir, "eval_reviews.txt")

baseline_dir = os.path.join(eval_dir, "baseline-bart")
path_20 = os.path.join(baseline_dir, "mask_sentences_20.txt")
path_40 = os.path.join(baseline_dir, "mask_sentences_40.txt")
path_60 = os.path.join(baseline_dir, "mask_sentences_60.txt")
path_80 = os.path.join(baseline_dir, "mask_sentences_80.txt")


def get_eval_lines(eval_path):
    print("Reading lines...")
    lines = open(eval_path, encoding='utf-8').read().strip().split('\n')
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

        # output_20 = evaluate_input(masked_sentences[0])
        # output_40 = evaluate_input(masked_sentences[1])
        # output_60 = evaluate_input(masked_sentences[2])
        # output_80 = evaluate_input(masked_sentences[3])

        mask_rate_20 = mask_rates[0]
        mask_rate_40 = mask_rates[1]
        mask_rate_60 = mask_rates[2]
        mask_rate_80 = mask_rates[3]

        ##For news headlines dataset:
        # f_20.write(masked_sentences[0] + '\t' + str(round(mask_rate_20, 3)) + '\n')
        # f_40.write(masked_sentences[1] + '\t' + str(round(mask_rate_40, 3)) + '\n')
        # f_60.write(masked_sentences[2] + '\t' + str(round(mask_rate_60, 3)) + '\n')
        # f_80.write(masked_sentences[3] + '\t' + str(round(mask_rate_80, 3)) + '\n')

        ##For reviews:
        f_20.write(masked_sentences[0] + '\t' + str(round(mask_rate_20, 3)) + '\t' + line[2] + '\n')
        f_40.write(masked_sentences[1] + '\t' + str(round(mask_rate_40, 3)) + '\t' + line[2] + '\n')
        f_60.write(masked_sentences[2] + '\t' + str(round(mask_rate_60, 3)) + '\t' + line[2] + '\n')
        f_80.write(masked_sentences[3] + '\t' + str(round(mask_rate_80, 3)) + '\t' + line[2] + '\n')

        counter += 1

    f_20.close()
    f_40.close()
    f_60.close()
    f_80.close()


write_results(eval_lines, path_20, path_40, path_60, path_80, 1)

