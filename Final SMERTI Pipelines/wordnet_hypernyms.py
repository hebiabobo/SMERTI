from nltk.corpus import wordnet
# ## Keyword Extraction & POS Tagging

# Import Stanford POS Tagger
from nltk.tag.stanford import StanfordPOSTagger

_path_to_model = r'E:\stanford\stanford-postagger-2018-10-16\models\english-bidirectional-distsim.tagger'
_path_to_jar = r'E:\stanford\stanford-postagger-2018-10-16\stanford-postagger.jar'
st = StanfordPOSTagger(model_filename=_path_to_model, path_to_jar=_path_to_jar)

# Install RAKE for keyword extraction
import re
import RAKE

Rake = RAKE.Rake(r"E:\stanford\stopwords.txt")

# def extract_keywords_and_POS(prompt):  # 提取句子中的关键词及其词性
#     POS_dict = {}
#     try:
#         tagged_prompt = st.tag(prompt.split())
#     except:
#         print("ERROR PROMPT: ", prompt)
#         return False
#     else:  # 无异常时执行
#         for pair in tagged_prompt:
#             POS_dict[pair[0]] = pair[1]
#         keywords_dict = {}
#         # format: Rake.run(prompt, minCharacters = X, maxWords = Y, minFrequency = Z)
#         keywords = Rake.run(prompt)
#         for pair in keywords:
#             words = pair[0].split()  # 可能是一个keyword可能有多个单词
#             for word in words:
#                 try:
#                     keywords_dict[word] = POS_dict[word]
#                 except:
#                     pass
#         return keywords_dict
#
#
# def get_coordinates(word, pos):
#     coordinates_lst = []
#     try:
#         word_sense = wordnet.synsets(word, pos)[0]
#         print("word_sense")
#         print(word_sense)
#     except:
#         try:
#             word_sense = wordnet.synsets(word)[0]
#         except:
#             return coordinates_lst
#     hypers = word_sense.hypernyms()
#     # hypers_sense = lambda s: s.hypernyms()
#     # hypers0 = list(word_sense.closure(hypers_sense))  # 这其实是求的传递闭包
#     # print("hypers")
#     # print(hypers)
#     # print("hypers0")
#     # print(hypers0)
#     lemma_count_max = 0
#     lemma_count_max_name = ""
#     for hyper in hypers:
#         # print("hyper")
#         # print(hyper)
#         coordinates = hyper.hyponyms()  # 对每个上义词, 找出其下义词
#         for coordinate in coordinates:
#             # print("coordinate")
#             # print(coordinate)
#             for l in coordinate.lemmas():  # 各种词形变换
#                 if (l.name().lower() != word) and (l.count() > lemma_count_max):  # 如果上义词的某个词形不是原词word，就添加进候选上义词dict中
#                     lemma_count_max = l.count()
#                     lemma_count_max_name = l.name().lower()
#     coordinates_lst.append(lemma_count_max_name)
#     return list(dict.fromkeys(coordinates_lst))
#
#
# # Example:
# # print(222)
# print(get_coordinates("person", "n"))
#
#
#
#
#
# def single_prompt_helper(keywords_lst, keywords_dict, fnc, chosen_nums):
#     counter = 1
#     chosen_keywords_lst = []
#     chosen_replacements_lst = []
#     print("keywords_lst")
#     print(keywords_lst)
#     for i in range(0, len(keywords_lst)):
#         if counter <= max(chosen_nums):
#             keyword = keywords_lst[i]
#             print("keyword")
#             print(keyword)
#             keyword_pos = keywords_dict[keyword][0].lower()
#             if keyword_pos == 'j':
#                 keyword_pos = 'a'
#             candidates = fnc(keyword, keyword_pos)  # 调用get_coordinates()获取coordinate
#             print("candidates")
#             print(candidates)
#             if len(candidates) != 0:
#                 counter += 1
#                 chosen_keywords_lst.append(keyword)
#                 chosen_replacement = candidates[0]  # 选第一个
#                 print("chosen_replacement")
#                 print(chosen_replacement)
#                 chosen_replacements_lst.append(chosen_replacement)
#         else:  # 在句中找出的keyword比给定允许的最多keyword多了, 那么多的就不要了
#             return chosen_keywords_lst, chosen_replacements_lst
#     return chosen_keywords_lst, chosen_replacements_lst
#
#
# def single_prompt_wordnet(input_sentence, nums_lst):
#     original_sentence = input_sentence
#     coordinates_sentence_lst = []
#     keywords_dict = extract_keywords_and_POS(input_sentence)
#     print("keywords_dict")  # key为关键词, value为词性的列表
#     print(keywords_dict)
#     if keywords_dict == False:
#         return []
#     keywords_lst = list(keywords_dict.keys())
#     print("keywords_lst")  # 获得关键词列表
#     print(keywords_lst)
#     num_keywords = len(keywords_lst)
#     sentence_coordinate = original_sentence
#     chosen_keywords, chosen_coordinates = single_prompt_helper(keywords_lst, keywords_dict, get_coordinates, nums_lst)
#     print("chosen_keywords")  # OE
#     print(chosen_keywords)
#     print("chosen_coordinates")  # RE
#     print(chosen_coordinates)
#     counter = 1
#     for chosen_word, chosen_coordinate in zip(chosen_keywords, chosen_coordinates):
#         sentence_coordinate = re.sub(r"\b%s\b" % chosen_word, chosen_coordinate, sentence_coordinate)  # 把句子中的OE换成RE
#         print("sentence_coordinate")
#         print(sentence_coordinate)
#         if counter in nums_lst:
#             coordinates_sentence_lst.append(re.sub('_', ' ', sentence_coordinate))  # 把句子中(的RE中)的下划线换成空格
#         counter += 1
#         print("coordinates_sentence_lst")
#         print(coordinates_sentence_lst)
#     return coordinates_sentence_lst


sentiment_words_list = []
f = open(r'E:\sentiment_words\Sentiment-Lexicon-master\sentiment_word.txt', encoding='utf-8')
for line in f:
    sentiment_words_list.append(line.strip())


def extract_keywords_and_POS(prompt, donot_sentiment):  # 提取句子中的关键词及其词性
    POS_dict = {}
    try:
        tagged_prompt = st.tag(prompt.split())
        # print("tagged_prompt")  # 句中所有的词及其词性的元组的list
        # print(tagged_prompt)
    except:
        print("ERROR PROMPT: ", prompt)
        return False
    else:  # 无异常时执行
        for pair in tagged_prompt:
            POS_dict[pair[0]] = pair[1]
        # print("POS_dict")  # 句中所有的词及其词性的dict
        # print(POS_dict)
        keywords_dict = {}
        # format: Rake.run(prompt, minCharacters = X, maxWords = Y, minFrequency = Z)
        keywords = Rake.run(prompt)  # 句中关键词及分数的元组的list
        # print("keywords")
        # print(keywords)
        for pair in keywords:
            # print("pair")  # 每个关键词及分数的元组
            # print(pair)
            words = pair[0].split()  # 一个keyword可能有多个单词
            # print("words")
            # print(words)
            for word in words:
                # print("word")
                # print(word)
                if donot_sentiment == 0:  # 消除重大情感词
                    if word not in sentiment_words_list:  # 如果关键词不在重大情感词列表里
                        keywords_dict[word] = POS_dict[word]  # 记录词和词性
                        # print("keywords_dict[word]")
                        # print(keywords_dict[word])
                else:
                    keywords_dict[word] = POS_dict[word]  # 记录词和词性
        return keywords_dict


def get_hypers(word, pos):
    coordinates_lst = []
    print("word, pos:---", word, pos)
    try:
        word_sense = wordnet.synsets(word, pos)[0]
        # print("word_sense")
        # print(word_sense)
    except:
        try:
            word_sense = wordnet.synsets(word)[0]
        except:  # wordnet中没有这个词
            return word, -1
    hypers = word_sense.hypernyms()
    # print("hypers", hypers)
    # hypers_sense = lambda s: s.hypernyms()
    # hypers0 = list(word_sense.closure(hypers_sense))  # 这其实是求的传递闭包
    lemma_count_max = 0
    lemma_count_max_name = ""
    for hyper in hypers:
        print("hyper")
        print(hyper)

        for l in hyper.lemmas():  # 各种词形变换
            if word not in l.name().lower() and l.name().lower() not in word and l.count() >= lemma_count_max:  # 如果上义词的某个词形不是原词word，就添加进候选上义词dict中
                lemma_count_max = l.count()
                lemma_count_max_name = l.name().lower()
    # print("lemma_count_max_name", lemma_count_max_name, "lemma_count_max", lemma_count_max)
    return lemma_count_max_name, lemma_count_max


def single_prompt_helper(keywords_lst, keywords_dict, fnc):
    final_keyword = ""
    candidate_max_count = 0
    final_candidate = ""
    print("keywords_lst")
    print(keywords_lst)
    print("keywords_dict")
    print(keywords_dict)
    for i in range(0, len(keywords_lst)):
        # if counter <= max(chosen_nums):
        keyword = keywords_lst[i]
        print("keyword")
        print(keyword)
        keyword_pos = keywords_dict[keyword][0].lower()  # 取词性的第一个字母的小写
        print("keyword_pos")
        print(keyword_pos)
        if keyword_pos == 'j':
            keyword_pos = 'a'
        candidate, candidate_count = fnc(keyword, keyword_pos)  # 调用get_coordinates()获取每一个keyword的coordinate
        print("candidate", candidate, "candidate_count", candidate_count)
        if candidate != "":
            if candidate_count >= candidate_max_count:
                candidate_max_count = candidate_count
                final_keyword = keyword
                final_candidate = candidate
        print("i")
        print(i)
    print("final_keyword:", final_keyword, "final_candidate:", final_candidate)
    return final_keyword, final_candidate


def single_prompt_wordnet(input_sentence):
    original_sentence = input_sentence
    do_not_care_sentiment = 0  # 默认不考虑情感词作为keyword
    keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment)
    # print("keywords_dict")  # key为关键词, value为词性的列表
    # print(keywords_dict)

    if keywords_dict:
        keywords_lst = list(keywords_dict.keys())
        # print("keywords_lst")  # 获得关键词列表
        # print(keywords_lst)
        num_keywords = len(keywords_lst)
        sentence_coordinate = original_sentence
        chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_hypers)
        if not chosen_keyword:  # 如果找不到候选的coordinate
            do_not_care_sentiment = 1  # 考虑情感词作为keyword
            keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment)
            keywords_lst = list(keywords_dict.keys())
            num_keywords = len(keywords_lst)
            sentence_coordinate = original_sentence
            chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_hypers)
    else:  # 没有keyword
        do_not_care_sentiment = 1  # 考虑情感词作为keyword
        keywords_dict = extract_keywords_and_POS(input_sentence, do_not_care_sentiment)
        keywords_lst = list(keywords_dict.keys())
        sentence_coordinate = original_sentence
        chosen_keyword, chosen_coordinate = single_prompt_helper(keywords_lst, keywords_dict, get_hypers)

    print("chosen_keyword")  # OE
    print(chosen_keyword)
    print("chosen_coordinate")  # RE
    print(chosen_coordinate)
    # counter = 1
    # for chosen_word, chosen_coordinate in zip(chosen_keywords, chosen_coordinates):
    sentence_coordinate = re.sub(r"\b%s\b" % chosen_keyword, chosen_coordinate, sentence_coordinate)  # 把句子中的OE换成RE
    # print("sentence_coordinate")
    # print(sentence_coordinate)
    return chosen_keyword, chosen_coordinate, sentence_coordinate


# Example:
nums_lst = [1, 2, 3]  # 最多选3个keyword, 最后选三个coordinate, 也就是生成三个新句子
# prompt = "an immortal being is explaining"
# prompt = "i love this place! very nice people running the cafe and the food is always good. stars!"
# prompt = "banned book week books that were challenged for having lgbt content"
prompt = 'this angelina jolie criticism might be the most unfair yet'
# hypers_lst = single_prompt_wordnet(prompt, nums_lst)
OE, RE, hypers_lst = single_prompt_wordnet(prompt)
print(hypers_lst)
