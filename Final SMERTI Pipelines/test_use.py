import tensorflow_hub as hub
import numpy as np
import RAKE
import re
import os
from string import punctuation
from collections import OrderedDict
from nltk.tree import *
from nltk.corpus import wordnet
from nltk.tag.stanford import StanfordPOSTagger
from stanfordcorenlp import StanfordCoreNLP
from transformers import BartTokenizer, BartForConditionalGeneration

nlp = StanfordCoreNLP(r'E:\stanfordcorenlp\stanford-corenlp-full-2018-02-27')

# USE句嵌入
embed = hub.load(r"E:\tensorflow_hub_objects\universal-sentence-encoder-large_5")

# Get Pretrained Bart
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# ## Keyword Extraction & POS Tagging

# Import Stanford POS Tagger

_path_to_model = r'E:\stanford\stanford-postagger-2018-10-16\models\english-bidirectional-distsim.tagger'
_path_to_jar = r'E:\stanford\stanford-postagger-2018-10-16\stanford-postagger.jar'
st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Install RAKE for keyword extraction

Rake = RAKE.Rake(r"E:\stanford\stopwords.txt")


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


def get_similarity_matrix(input_list):
    # message_embeddings = session.run(
    #     embedded_text, feed_dict={text_input: input_list})
    message_embeddings = embed(input_list)
    corr = np.inner(message_embeddings, message_embeddings)
    return corr


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


def get_full_index_max(full_input_list, replaced_entity):
    full_index_max = full_input_list.index(replaced_entity)
    return full_index_max


def get_original_similarity(similarity_matrix, index_max):
    original_similarity = similarity_matrix[index_max]
    return original_similarity


def get_new_length(new_sentence):
    new_word_list = sentence_to_words(new_sentence)
    new_list_len = len(list(filter(lambda x: x not in punctuation, new_word_list)))
    return new_list_len


def get_masked_sentence(mask_threshold, similarity_threshold, length, similarity_vector_old, word_list,
                        index_max, original_len, similar_words, new_sentence, mask_sentence, replacement_entity):
    # length表示目前句中所有与OE相似的entities的单词数(masked的单词数), 1000啥也不是
    if length <= int(round((original_len * mask_threshold))):  # 如果当前masked的单词数小于等于句中所允许mask的最大单词数了
        return mask_sentence, length / original_len  # 则返回最终SMM后的句子以及本句的实际mask率

    else:  # 如果当前masked的单词数大于句中所允许mask的最大单词数了
        counter = 0
        indices_list = []  # 和OE的similarity之间满足ST的entity的index

        for score in similarity_vector_old:  # OE和原句中所有entities的similarity
            # 如果OE和原句中当前entity(且不是OE本身)的similarity大于ST, 添加index
            if score > similarity_threshold and counter != index_max and counter not in indices_list:
                indices_list.append(counter)
            counter += 1

        print("indices_list")
        print(indices_list)
        similar_words = []  # 和OE的similarity之间满足ST的entity
        for index in indices_list:
            similar_words.append(word_list[index])
        similar_words = list(filter(lambda x: x not in punctuation, similar_words))  # 过滤掉标点
        similar_words = list(set(similar_words))
        similar_words.sort(key=lambda x: len(x.split()), reverse=True)  # entity中包含的单词个数多的排在前面
        print("similar_words")
        print(similar_words)

        temp_mask_num, temp_mask_sentence = mask_similar_words(similar_words, new_sentence, replacement_entity)
        print("temp_mask_num")  # 句中所有与OE相似的entities的单词数(masked的单词数)
        print(temp_mask_num)
        print("temp_mask_sentence")  # SMM后的句子
        print(temp_mask_sentence)

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
            # print("word")
            # print(word)
            if word not in replacement_entity and replacement_entity not in word:
                # print(33333)
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


def mask_fnc(original_similarity, full_input_list, index_max, original_len, new_sentence,
             mask_threshold, similarity_threshold, replacement_entity):
    masked_sentence, final_mask_rate = get_masked_sentence(mask_threshold, similarity_threshold, 1000,
                                                           original_similarity[:-1], full_input_list, index_max,
                                                           original_len, [], new_sentence, "", replacement_entity)
    # print("mask_fnc :masked_sentence")
    # print(masked_sentence)
    masked_word_list = masked_sentence.split()
    masked_group_list = mask_groupings(masked_word_list)
    masked_group_sentence = ' '.join(masked_group_list)
    return masked_group_sentence, final_mask_rate


def main_USE_function(initial_sentence, new_sentence, original_entity, replacement_entity):
    # initial_sentence = process_text(input_sentence)
    # replacement_entity = process_text(replacement_entity)
    word_list = get_word_list(initial_sentence)
    original_len = get_original_len(word_list)

    leaves = get_trees(initial_sentence, replacement_entity)
    entities_list = create_entities_list(leaves)

    if not entities_list:
        entities_list = word_list

    full_input_list = create_input_list(entities_list, word_list, replacement_entity)
    print("full_input_list")
    print(full_input_list)
    OE_index = get_full_index_max(full_input_list, original_entity)
    # print("OE_index")
    # print(OE_index)
    similarity_matrix = get_similarity_matrix(full_input_list)
    original_similarity = get_original_similarity(similarity_matrix, OE_index)
    # print("original_similarity")
    # print(original_similarity)

    masked_group_sentence_1, final_mask_rate_1 = mask_fnc(original_similarity, full_input_list,
                                                          OE_index, original_len, new_sentence, 0.2, 0.4,
                                                          replacement_entity)
    masked_group_sentence_2, final_mask_rate_2 = mask_fnc(original_similarity, full_input_list,
                                                          OE_index, original_len, new_sentence, 0.4, 0.3,
                                                          replacement_entity)
    masked_group_sentence_3, final_mask_rate_3 = mask_fnc(original_similarity, full_input_list,
                                                          OE_index, original_len, new_sentence, 0.6, 0.2,
                                                          replacement_entity)
    masked_group_sentence_4, final_mask_rate_4 = mask_fnc(original_similarity, full_input_list,
                                                          OE_index, original_len, new_sentence, 0.8, 0.1,
                                                          replacement_entity)
    masked_group_sentences = [masked_group_sentence_1, masked_group_sentence_2, masked_group_sentence_3,
                              masked_group_sentence_4]
    final_mask_rates = [final_mask_rate_1, final_mask_rate_2, final_mask_rate_3, final_mask_rate_4]

    return masked_group_sentences, final_mask_rates


user_input = 'quincy jones apologizes for giving best interviews of the year'
OE = 'interviews'
RE = 'interview'
RE_sentence = 'quincy jones apologizes for giving best interview of the year'
masked_sentences, mask_rates = main_USE_function(user_input, RE_sentence, OE, RE)
