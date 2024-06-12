import os
import tensorflow_hub as hub
import numpy as np

embed = hub.load(r"E:\tensorflow_hub_objects\universal-sentence-encoder-large_5")


def get_similarity_score(input_list):
    # message_embeddings = session.run(
    #     embedded_text, feed_dict={text_input: input_list})
    # print("input_list")  # list中的第一个str为RE,第二个str为RE_sentence
    # print(input_list)
    # print("input_list")
    # print(input_list)
    message_embeddings = embed(input_list)  # 对RE和RE_sentence分别句嵌入,格式为[[...],[...]]
    # print("message_embeddings")
    # print(message_embeddings)
    corr = np.inner(message_embeddings, message_embeddings)  # 计算内积矩阵
    # print("corr")
    # print(corr)
    sim_score = corr[0][1]  # 矩阵的[0][1]位置就是它的相似度
    # print("sim_score")
    # print(sim_score)
    # print("Similarity score for {}: ".format(input_list), sim_score, '\n')
    return sim_score


# Get average CSS of the original text
def get_input_CSS(input_path, if_baseline):
    print("\n\nCurrently calculating scores for file: ", input_path)
    counter = 1
    scores_lst = []
    input_lines = [line.strip().split('\t') for line in open(input_path, encoding='utf-8')]
    for line in input_lines:
        # print("Evaluating line: ", counter)
        if if_baseline:
            sim_score = get_similarity_score([line[0], line[1]])  # list中的第一个str为RE,第二个str为RE_sentence
        else:
            # For Reviews
            sim_score = get_similarity_score([line[3], line[1]])  # list中的第一个str为RE,第二个str为RE_sentence
            # For News
            # sim_score = get_similarity_score([line[2], line[1]])
        # print("sim_score   " + str(counter) + "   :    " + str(sim_score))
        scores_lst.append(sim_score)
        counter += 1
    avg_score = np.mean(scores_lst)
    # print("Average score(RE and OE_sentence): ", avg_score)
    return avg_score


# Get average CSS of generated text by model(s)
def get_output_CSS(input_path, output_path, if_baseline):
    print("\n\nCurrently calculating scores for file: ", output_path)
    counter = 1
    scores_lst = []
    input_lines = [line.strip().split('\t') for line in open(input_path, encoding='utf-8')]
    output_lines = [line.strip().split('\t') for line in open(output_path, encoding='utf-8')]
    for input_line, output_line in zip(input_lines, output_lines):
        # print("Evaluating line: ", counter)
        if if_baseline:
            sim_score = get_similarity_score([input_line[0], output_line[0]])
        else:
            # For Reviews
            sim_score = get_similarity_score([input_line[3], output_line[0]])
            # For News
            # sim_score = get_similarity_score([input_line[2], output_line[0]])

        # print("sim_score   " + str(counter) + "   :    " + str(sim_score))
        scores_lst.append(sim_score)
        counter += 1
    avg_score = np.mean(scores_lst)
    # print("Average score(RE and RE_sentence): ", avg_score)
    return avg_score


# original_lines = 3000
# output_line = 2993
#
# ste_dir = r"E:\PycharmProjects\SMERTI-master"
# dataset_dir = os.path.join(ste_dir, "Amazon_Dataset")
# eval_dir = os.path.join(dataset_dir, "evaluation_data")
# # method_dir = os.path.join(eval_dir, "ste_bart_mutimask")
# method_dir = os.path.join(eval_dir, "ste_bart_large_mutimask")

# For SPA, BLEU and CSS
# output_lines_num = 2993
# input_path = os.path.join(method_dir, "reviews_filtered_input.txt")
# output_path = os.path.join(method_dir, "reviews_output_80.txt")

# CSS
# input_CSS = get_input_CSS(input_path)
# output_CSS = get_output_CSS(input_path, output_path)


def choose_dataset(dataset_name, method_name):
    If_baseline = False

    if dataset_name == 'News':
        eval_dir = r"E:\PycharmProjects\SMERTI-master\News_Dataset\evaluation_data"
        original_path = os.path.join(eval_dir, "eval_headlines.txt")
        if method_name == 'baseline_result':
            method_dir = os.path.join(eval_dir, "baseline_result")
            input_path = original_path
            If_baseline = True
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
            If_baseline = True
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
            If_baseline = True
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
            If_baseline = True
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
            If_baseline = True
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
            If_baseline = True
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
            If_baseline = True
        elif method_name == 'baseline-bart':
            method_dir = os.path.join(eval_dir, "baseline-bart")
            input_path = original_path
            If_baseline = True
        elif method_name == 'baseline-bart-large':
            method_dir = os.path.join(eval_dir, "baseline-bart-large")
            input_path = original_path
            If_baseline = True
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

    input_CSS = get_input_CSS(input_path, If_baseline)
    print("Average score(RE and OE_sentence): ", input_CSS)
    output_CSS = get_output_CSS(input_path, output_path1, If_baseline)
    print("20:  Average score(RE and RE_sentence): ", output_CSS)
    output_CSS = get_output_CSS(input_path, output_path2, If_baseline)
    print("40:  Average score(RE and RE_sentence): ", output_CSS)
    output_CSS = get_output_CSS(input_path, output_path3, If_baseline)
    print("60:  Average score(RE and RE_sentence): ", output_CSS)
    output_CSS = get_output_CSS(input_path, output_path4, If_baseline)
    print("80:  Average score(RE and RE_sentence): ", output_CSS)


Dataset_name = 'Yelp'
# Method_name = 'ste-bart-mutimask-coordinate-similiar'
# Method_name = 'ste-bart-mutimask-hyper-similiar'
# Method_name = 'ste-bart-mutimask-hypo-similiar'
# Method_name = 'baseline-bart'
# Method_name = 'baseline-bart-large'
# Method_name = 'ste-bart-large-coordinate-similiar'
# Method_name = 'ste-bart-large-hyper-similiar'
Method_name = 'ste-bart-large-hypo-similiar'
# Method_name = 'baseline_result'
choose_dataset(Dataset_name, Method_name)

