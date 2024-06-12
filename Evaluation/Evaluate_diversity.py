import os
import numpy as np
# from metric_utils import *
from tqdm import tqdm
from nltk import word_tokenize
import sacrebleu


# sentence embedding
# embedder = SentenceTransformer('bert-base-nli-mean-tokens')


# def load_data(prompt_file, continuation_file):
#     print("Reading lines...")
#     prompts = []
#     prompt_f = open(prompt_file, 'r')
#     prompt_lines = prompt_f.readlines()
#     for prompt in prompt_lines:
#         prompts.append(prompt.strip('\n').strip('\ufeff'))
#     continuations = []
#     cont_f = open(continuation_file, 'r')
#     cont_lines = cont_f.readlines()
#     for cs in cont_lines:
#         conts = cs.strip('\n').strip('\ufeff').split(" <CAND_SEP> ")
#         continuations.append(conts)
#     assert len(prompts) == len(continuations)
#     print('Loaded: {}'.format(len(prompts)))
#     return prompts, continuations


# def evaluate_SBLEU(continuations):
#     all_results = []
#     for continuation in tqdm(continuations):
#         result = get_self_metric_corpus_parallel(continuation)
#         all_results.append(result)
#     final_result = np.average(all_results)
#     return final_result, all_results

def evaluate_UTR(lines):
    all_results = []
    for line in lines:
        # result = get_unique_trigrams([line[0]])
        # all_results.append(result)
        if len(line) < 3 or len(line[0]) == 0 or len(line[0].split()) == 0:
            print("Error output line: ", line)
            continue
        else:
            all_results.append(get_unique_trigrams([line[0]]))
    final_result = np.average(all_results)
    return final_result


def get_unique_trigrams(hyp_population):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # returns the unique trigram fraction in this population.
    # Higher the unique trigram fraction, more the diversity
    unique_trigrams = set()
    total_trigrams = 0

    for i, hyp_i in enumerate(hyp_population):
        hyp_i_words = hyp_i.strip().split()
        if len(hyp_i_words) >= 3:
            total_trigrams += len(hyp_i_words) - 2
            for j in range(len(hyp_i_words) - 2):
                trigram = " ".join(hyp_i_words[j:j + 2])
                unique_trigrams.add(trigram)

    unique_trigram_fraction = len(unique_trigrams) / (total_trigrams + 1e-10)
    if total_trigrams == 0:
        unique_trigram_fraction = 0.0
    return unique_trigram_fraction


def TTR_score(sentence):
    word_lst = word_tokenize(sentence)
    clean_word_lst = []

    for word in word_lst:
        clean_word_lst.append(word)

    unique_word_lst = set(clean_word_lst)
    TTR = len(unique_word_lst) / len(clean_word_lst)
    # print("Sentence: ", sentence, " / TTR: ", TTR)
    return TTR


def evaluate_TTR(sentences):
    all_results = []
    for sentence in sentences:
        # For reviews
        # if len(sentence) < 3 or len(sentence[0]) == 0 or len(sentence[0].split()) == 0:
        # For News and bartbaseline
        if len(sentence) < 2 or len(sentence[0]) == 0 or len(sentence[0].split()) == 0:
            print("Error output line: ", sentence)
            continue
        else:
            all_results.append(TTR_score(sentence[0]))
    final_result = np.average(all_results)
    # Below line is if condition for news headlines dataset
    # if len(output_line) < 2 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:

    return final_result


def evaluate_input_TTR(sentences):
    all_results = []
    for sentence in sentences:
        all_results.append(TTR_score(sentence[1]))
    final_result = np.average(all_results)
    # Below line is if condition for news headlines dataset
    # if len(output_line) < 2 or len(output_line[0]) == 0 or len(output_line[0].split()) == 0:

    return final_result


def get_corpus_bleu_parallel(hyp_list,refs_list):
    hyp_lines = [hyp.strip() for hyp in hyp_list]
    ref_lines_list = [[ref.strip() for ref in refs] for refs in refs_list]

    ref_size = len(ref_lines_list[0])

    ref_structure = [[] for i in range(ref_size)]
    for i in range(ref_size):
        ref_structure[i] = [ref_lines_list[j][i] for j in range(len(ref_lines_list))]

    bleu_score = sacrebleu.corpus_bleu(hyp_lines,ref_structure,smooth_method='exp')

    return bleu_score


def get_self_metric_corpus_parallel(hyp_population, metric="bleu_score", bert_scorer=None):
    # hyp_population: list of line strings, each string is a hypothesis/continuation
    # metric: Choice of closeness metric, bert_score or bleu_score
    # returns the self-metric [where metric is a closeness metric i.e bert_score or bleu_score]
    # Lower the self-metric, more the diversity
    closest_metrics = []

    # hyps = []
    rest_of_corpuses = []

    hyps = [hyp_i for hyp_i in hyp_population]
    # rest_of_corpuses = [hyp_population[:-1] if i == (len(hyp_population)-1) else (hyp_population[:i] + hyp_population[i+1:]) for i in range(len(hyp_population))]

    # for i,hyp_i in enumerate(hyp_population):
    for i in range(len(hyp_population)):
        rest_of_corpus = None

        if i == len(hyp_population) - 1:
            rest_of_corpus = hyp_population[:-1]
        else:
            rest_of_corpus = hyp_population[:i] + hyp_population[i + 1:]

        # hyps.append(hyp_i)
        rest_of_corpuses.append(rest_of_corpus)

    self_metric_score = get_corpus_bleu_parallel(hyps, rest_of_corpuses).score / len(rest_of_corpuses[0])

    return self_metric_score


def evaluate_SBLEU(lines):
    all_results = []
    for line in lines:
        result = get_self_metric_corpus_parallel(line[0])
        all_results.append(result)
    final_result = np.average(all_results)
    return final_result, all_results


def choose_dataset(dataset_name, method_name):
    original_lines_num = 3000

    if dataset_name == 'News':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_news.txt")
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

    input_lines = [line.strip().split('\t') for line in open(input_path, encoding='utf-8')]
    output_lines1 = [line.strip().split('\t') for line in open(output_path1, encoding='utf-8')]
    output_lines2 = [line.strip().split('\t') for line in open(output_path2, encoding='utf-8')]
    output_lines3 = [line.strip().split('\t') for line in open(output_path3, encoding='utf-8')]
    output_lines4 = [line.strip().split('\t') for line in open(output_path4, encoding='utf-8')]

    TTR = evaluate_input_TTR(input_lines)
    print("Input TTR score : ", TTR)
    TTR = evaluate_TTR(output_lines1)
    print("Final TTR 20score : ", TTR)
    TTR = evaluate_TTR(output_lines2)
    print("Final TTR 40score : ", TTR)
    TTR = evaluate_TTR(output_lines3)
    print("Final TTR 60score : ", TTR)
    TTR = evaluate_TTR(output_lines4)
    print("Final TTR 80score : ", TTR)
    # UTR = evaluate_UTR(output_lines1)
    # print("Final UTR 20score : ", UTR)
    # UTR = evaluate_UTR(output_lines2)
    # print("Final UTR 40score : ", UTR)
    # UTR = evaluate_UTR(output_lines3)
    # print("Final UTR 60score : ", UTR)
    # UTR = evaluate_UTR(output_lines4)
    # print("Final UTR 80score : ", UTR)


Dataset_name = 'News'
Method_name = 'ste-bart-mutimask-coordinate-similiar'
# Method_name = 'baseline_result'
# Method_name = 'ste-bart-mutimask-hyper-similiar'
# Method_name = 'ste-bart-mutimask-hypo-similiar'
# Method_name = 'baseline-bart'
# Method_name = 'baseline-bart-large'
# Method_name = 'ste-bart-large-coordinate-similiar'
# Output_lines_num = 2993
# Method_name = 'ste-bart-large-hyper-similiar'
# Method_name = 'ste-bart-large-hypo-similiar'
choose_dataset(Dataset_name, Method_name)


