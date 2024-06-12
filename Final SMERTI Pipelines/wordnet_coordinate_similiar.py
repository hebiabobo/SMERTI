from nltk.corpus import wordnet
import tensorflow_hub as hub
# ## Keyword Extraction & POS Tagging

# Import Stanford POS Tagger
from nltk.tag.stanford import StanfordPOSTagger

_path_to_model = r'E:\stanford\stanford-postagger-2018-10-16\models\english-bidirectional-distsim.tagger'
_path_to_jar = r'E:\stanford\stanford-postagger-2018-10-16\stanford-postagger.jar'
st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Install RAKE for keyword extraction
import re
import RAKE
import numpy as np

Rake = RAKE.Rake(r"E:\stanford\stopwords.txt")
embed = hub.load(r"E:\tensorflow_hub_objects\universal-sentence-encoder-large_5")


def get_similarity_score(input_list):
    message_embeddings = embed(input_list)  # 对RE和RE_sentence分别句嵌入,格式为[[...],[...]]
    corr = np.inner(message_embeddings, message_embeddings)  # 计算内积矩阵
    sim_score = corr[0][1]  # 矩阵的[0][1]位置就是它的相似度
    # print("Similarity score for {}: ".format(input_list), sim_score, '\n')
    return sim_score


sentiment_words_list = []
f = open(r'E:\sentiment_words\Sentiment-Lexicon-master\sentiment_word.txt', encoding='utf-8')
for line in f:
    sentiment_words_list.append(line.strip())


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
        print("POS_dict")
        print(POS_dict)
        print("keywords")
        print(keywords)
        for pair in keywords:
            words = pair[0].split()  # 一个keyword可能有多个单词
            print("words")
            print(words)
            for word in words:
                print("word")
                print(word)
                try:
                    if POS_dict[word] != 'NN' and POS_dict[word] != 'NNS':  # 不考虑名词和名词复数以外的词
                        continue
                except:  # 如果原句中产生的dict里面没有这个词
                    continue
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
    lemma_keyword_sim_max = -1
    lemma_keyword_sim_max_name = ""
    for hyper in hypers:
        coordinates = hyper.hyponyms()  # 对每个上义词, 找出其下义词
        for coordinate in coordinates:
            for l in coordinate.lemmas():  # 各种词形变换
                if word not in l.name().lower() and l.name().lower() not in word:
                    sim_score = get_similarity_score([l.name().lower(), word])
                    # if l.count() >= lemma_count_max:  # 如果上义词的某个词形不是原词word，就添加进候选上义词dict中?
                    if sim_score >= lemma_keyword_sim_max:
                        lemma_keyword_sim_max = sim_score
                        lemma_keyword_sim_max_name = l.name().lower()
    return lemma_keyword_sim_max_name, lemma_keyword_sim_max


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
    print("keywords_dict")
    print(keywords_dict)

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


# Example:
nums_lst = [1, 2, 3]  # 最多选3个keyword, 最后选三个coordinate, 也就是生成三个新句子
# prompt = "an immortal being is explaining"
# prompt = "i love this place! very nice people running the cafe and the food is always good. stars!"
# prompt = "great shirt ! looks great , nice weathered look . thirteen more words required . well here are a few more amazon , hee hee !"
# prompt = "banned book week books that were challenged for having lgbt content"
# prompt = 'this angelina jolie criticism might be the most unfair yet'
# hypers_lst = single_prompt_wordnet(prompt, nums_lst)
prompt = "amazing and hystericalyou won't be wasting your time by watching it . i really enjoyed all of the characters . watch it ."
OE, RE, hypers_lst = single_prompt_wordnet(prompt)
print(hypers_lst)
