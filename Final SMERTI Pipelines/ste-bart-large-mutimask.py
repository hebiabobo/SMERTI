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
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# ## Keyword Extraction & POS Tagging

# Import Stanford POS Tagger

_path_to_model = r'E:\stanford\stanford-postagger-2018-10-16\models\english-bidirectional-distsim.tagger'
_path_to_jar = r'E:\stanford\stanford-postagger-2018-10-16\stanford-postagger.jar'
st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Install RAKE for keyword extraction

Rake = RAKE.Rake(r"E:\stanford\stopwords.txt")

# filter sentiment words

sentiment_words_list = []
f = open(r'E:\sentiment_words\Sentiment-Lexicon-master\sentiment_word.txt', encoding='utf-8')
for line in f:
    sentiment_words_list.append(line.strip())


# Get RE


def extract_keywords_and_POS(prompt, donot_sentiment, do_all):  # 提取句子中的关键词及其词性
    POS_dict = {}
    try:
        tagged_prompt = st.tag(prompt.split())  # 句中所有的词及其词性的元组的list
    except:
        print("ERROR PROMPT: ", prompt)
        return False
    else:  # 无异常时执行
        for pair in tagged_prompt:
            POS_dict[pair[0]] = pair[1]  # 句中所有的词及其词性的dict
        keywords_dict = {}
        if do_all == 1:
            return POS_dict
        # format: Rake.run(prompt, minCharacters = X, maxWords = Y, minFrequency = Z)
        keywords = Rake.run(prompt)  # 句中关键词及分数的元组的list
        for pair in keywords:
            words = pair[0].split()  # 一个keyword可能有多个单词
            for word in words:
                if donot_sentiment == 0:  # 消除重大情感词
                    if word not in sentiment_words_list:  # 如果关键词不在重大情感词列表里
                        try:
                            keywords_dict[word] = POS_dict[word]  # 记录词和词性
                        except KeyError:
                            continue
                else:
                    try:
                        keywords_dict[word] = POS_dict[word]  # 记录词和词性
                    except KeyError:
                        continue
        return keywords_dict


def get_coordinates(word, pos):
    try:
        word_sense = wordnet.synsets(word, pos)[0]
    except:
        try:
            word_sense = wordnet.synsets(word)[0]
        except:
            return word, -1
    hypers = word_sense.hypernyms()
    # hypers_sense = lambda s: s.hypernyms()
    # hypers0 = list(word_sense.closure(hypers_sense))  # 这其实是求的传递闭包
    lemma_count_max = 0
    lemma_count_max_name = ""
    for hyper in hypers:
        coordinates = hyper.hyponyms()  # 对每个上义词, 找出其下义词
        for coordinate in coordinates:
            for lemma in coordinate.lemmas():  # 各种词形变换
                if word not in lemma.name().lower() and lemma.name().lower() not in word and lemma.count() >= lemma_count_max:
                    # 如果上义词的某个词形不是原词word，就添加进候选上义词dict中
                    lemma_count_max = lemma.count()
                    lemma_count_max_name = lemma.name().lower()
    # if lemma_count_max_name == "":
    #     for hyper in hypers:
    #         # print("hyper")
    #         # print(hyper)
    #         coordinates = hyper.hyponyms()  # 对每个上义词, 找出其下义词
    #         for coordinate in coordinates:
    #             print("coordinate")
    #             print(coordinate)
    #             for lemma in coordinate.lemmas():  # 各种词形变换
    #                 if word != lemma.name().lower() and lemma.count() >= lemma_count_max:  # 允许词形变换
    #                     lemma_count_max = lemma.count()
    #                     lemma_count_max_name = lemma.name().lower()
    return lemma_count_max_name, lemma_count_max


def single_prompt_helper(keywords_lst, keywords_dict, fnc):
    final_keyword = ""
    candidate_max_count = 0
    final_candidate = ""
    for i in range(0, len(keywords_lst)):
        keyword = keywords_lst[i]
        keyword_pos = keywords_dict[keyword][0].lower()
        if keyword_pos == 'j':
            keyword_pos = 'a'
        candidate, candidate_count = fnc(keyword, keyword_pos)  # 调用get_coordinates()获取每一个keyword的coordinate
        if candidate != "":
            if candidate_count >= candidate_max_count:
                candidate_max_count = candidate_count
                final_keyword = keyword
                final_candidate = candidate
    return final_keyword, final_candidate


def single_prompt_wordnet(input_sentence):
    original_sentence = input_sentence
    do_not_care_sentiment = 0  # 默认不考虑情感词作为keyword
    keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment, 0)  # key为关键词, value为词性的列表, 消除了情感词

    if keywords_dict:
        keywords_lst = list(keywords_dict.keys())  # 获得关键词列表
        sentence_coordinate = original_sentence
        chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_coordinates)
        if not chosen_keyword:  # 找不到coordinate
            do_not_care_sentiment = 1  # 考虑情感词作为keyword
            keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment, 0)
            keywords_lst = list(keywords_dict.keys())
            sentence_coordinate = original_sentence
            chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_coordinates)
    else:  # 在消除了情感词的keyword里没有其他keyword
        do_not_care_sentiment = 1  # 考虑情感词作为keyword
        keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment, 0)
        keywords_lst = list(keywords_dict.keys())
        sentence_coordinate = original_sentence
        chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_coordinates)
    if not chosen_keyword:
        do_not_care_sentiment = 1  # 考虑情感词作为keyword
        keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment, 1)
        keywords_lst = list(keywords_dict.keys())
        sentence_coordinate = original_sentence
        chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_coordinates)
    sentence_coordinate = re.sub(r"\b%s\b" % chosen_keyword, chosen_coordinate, sentence_coordinate)  # 把句子中的OE换成RE
    return chosen_keyword, chosen_coordinate, sentence_coordinate


# SMM


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

        similar_words = []  # 和OE的similarity之间满足ST的entity
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


def mask_fnc(original_similarity, full_input_list, index_max, original_len, new_sentence,
             mask_threshold, similarity_threshold, replacement_entity):
    masked_sentence, final_mask_rate = get_masked_sentence(mask_threshold, similarity_threshold, 1000,
                                                           original_similarity[:-1], full_input_list, index_max,
                                                           original_len, [], new_sentence, "", replacement_entity)

    return masked_sentence, final_mask_rate


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
    OE_index = get_full_index_max(full_input_list, original_entity)
    similarity_matrix = get_similarity_matrix(full_input_list)
    original_similarity = get_original_similarity(similarity_matrix, OE_index)

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


# ## Example

# In[ ]:


# Example (for news headlines)

# nums_lst = [1, 2, 3]  # 最多选3个keyword
# OE_sentence = "i love this place! very nice people running the cafe and the food is always good. stars!"
# OE, RE, RE_sentence = single_prompt_wordnet(OE_sentence)

#
# masked_sentences, mask_rates = main_USE_function(OE_sentence, RE_sentence, OE, RE)  # 生成masked句子
# print("\nFinal masked sentences: ", masked_sentences)
# print("Final mask rates: ", mask_rates, '\n')


# #Evaluation Data Preparation

# ## Write Pipeline Outputs for Evaluation Lines


def get_eval_lines(eval_path):
    print("Reading lines...")
    lines = open(eval_path, encoding='utf-8').read().strip().split('\n')
    eval_lines = [line.split('\t') for line in lines]
    return eval_lines


def bart_generation_mask_to_unmask(masked_sentence):
    if '<mask>' not in masked_sentence:
        return masked_sentence
    else:
        input_ids = tokenizer(masked_sentence, return_tensors='pt')['input_ids']
        generated_ids = model.generate(input_ids, max_length=50, repetition_penalty=5.0, do_sample=True, top_k=50,
                                       top_p=0.8)
        final_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return final_generation


def list_to_str(sentence_list):
    return "".join(sentence_list)


def write_results(eval_lines, filtered_input_path, path_20, path_40, path_60, path_80, counter_start, dataset_name):
    f_filter_input = open(filtered_input_path, 'a', encoding='utf-8')
    f_20 = open(path_20, 'a', encoding='utf-8')
    f_40 = open(path_40, 'a', encoding='utf-8')
    f_60 = open(path_60, 'a', encoding='utf-8')
    f_80 = open(path_80, 'a', encoding='utf-8')

    counter = counter_start
    success_counter = counter_start

    for line in eval_lines:
        print("\nCurrently evaluating line {}".format(counter))
        print("Line: ", line)  # 这是list不是str

        # replacement_entity = line[0]
        user_input = line[1]
        OE, RE, RE_sentence = single_prompt_wordnet(user_input)
        if not OE:
            print("Can't find coordinate!")
            counter += 1
            continue
        masked_sentences, mask_rates = main_USE_function(user_input, RE_sentence, OE, RE)
        print("masked_sentences")
        print(masked_sentences)

        output_20_list = bart_generation_mask_to_unmask(masked_sentences[0])
        output_40_list = bart_generation_mask_to_unmask(masked_sentences[1])
        output_60_list = bart_generation_mask_to_unmask(masked_sentences[2])
        output_80_list = bart_generation_mask_to_unmask(masked_sentences[3])

        output_20 = list_to_str(output_20_list)
        output_40 = list_to_str(output_40_list)
        output_60 = list_to_str(output_60_list)
        output_80 = list_to_str(output_80_list)

        mask_rate_20 = mask_rates[0]
        mask_rate_40 = mask_rates[1]
        mask_rate_60 = mask_rates[2]
        mask_rate_80 = mask_rates[3]

        if dataset_name == "News":
            # For news headlines dataset:
            f_filter_input.write(line[0] + '\t' + line[1] + '\t' + RE + '\n')
            f_20.write(output_20 + '\t' + str(round(mask_rate_20, 3)) + '\n')
            f_40.write(output_40 + '\t' + str(round(mask_rate_40, 3)) + '\n')
            f_60.write(output_60 + '\t' + str(round(mask_rate_60, 3)) + '\n')
            f_80.write(output_80 + '\t' + str(round(mask_rate_80, 3)) + '\n')

        else:
            # For reviews:
            f_filter_input.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + RE + '\n')
            f_20.write(output_20 + '\t' + str(round(mask_rate_20, 3)) + '\t' + line[2] + '\n')
            f_40.write(output_40 + '\t' + str(round(mask_rate_40, 3)) + '\t' + line[2] + '\n')
            f_60.write(output_60 + '\t' + str(round(mask_rate_60, 3)) + '\t' + line[2] + '\n')
            f_80.write(output_80 + '\t' + str(round(mask_rate_80, 3)) + '\t' + line[2] + '\n')

        counter += 1
        success_counter += 1
        # if counter > 2:
        #     break

    f_filter_input.close()
    f_20.close()
    f_40.close()
    f_60.close()
    f_80.close()
    print("success_counter")
    print(success_counter - 1)


def choose_dataset(Dataset_name):
    if Dataset_name == 'News':
        # For news
        eval_dir = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data"
        eval_path = os.path.join(eval_dir, "eval_headlines_(1).txt")
        eval_lines = get_eval_lines(eval_path)

        path_mutimask = os.path.join(eval_dir, "ste_bart_large_mutimask")
        filtered_input_path = os.path.join(path_mutimask, "news_filtered_input.txt")
        path_20 = os.path.join(eval_dir, "news_output_20.txt")
        path_40 = os.path.join(eval_dir, "news_output_40.txt")
        path_60 = os.path.join(eval_dir, "news_output_60.txt")
        path_80 = os.path.join(eval_dir, "news_output_80.txt")
    elif Dataset_name == 'Amazon':
        # For Amazon reviews
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Amazon_Dataset\evaluation_data"
        eval_path = os.path.join(eval_dir, "eval_reviews.txt")
        eval_lines = get_eval_lines(eval_path)

        path_mutimask = os.path.join(eval_dir, "ste_bart_large_mutimask")
        filtered_input_path = os.path.join(path_mutimask, "reviews_filtered_input.txt")
        path_20 = os.path.join(path_mutimask, "reviews_output_20.txt")
        path_40 = os.path.join(path_mutimask, "reviews_output_40.txt")
        path_60 = os.path.join(path_mutimask, "reviews_output_60.txt")
        path_80 = os.path.join(path_mutimask, "reviews_output_80.txt")
    elif Dataset_name == 'Yelp':
        # For Yelp reviews
        eval_dir = r"E:\PycharmProjects\SMERTI-master\Yelp_Dataset\evaluation_data"
        eval_path = os.path.join(eval_dir, "eval_reviews_(1).txt")
        eval_lines = get_eval_lines(eval_path)

        path_mutimask = os.path.join(eval_dir, "ste_bart_large_mutimask")
        filtered_input_path = os.path.join(path_mutimask, "reviews_filtered_input.txt")
        path_20 = os.path.join(path_mutimask, "reviews_output_20.txt")
        path_40 = os.path.join(path_mutimask, "reviews_output_40.txt")
        path_60 = os.path.join(path_mutimask, "reviews_output_60.txt")
        path_80 = os.path.join(path_mutimask, "reviews_output_80.txt")
    else:
        print("Wrong Dataset Name !")
        return

    write_results(eval_lines, filtered_input_path, path_20, path_40, path_60, path_80, 1, Dataset_name)


choose_dataset('Amazon')






