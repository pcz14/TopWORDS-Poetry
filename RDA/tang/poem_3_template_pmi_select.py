# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:49
# @Author  : Changzai Pan
# @File    : wiki_topwords.py

from RDA.tang.poem_0_config import *
import random
from utils_seg import read_txt_encoding, check_result_dir, integrate_result
from collections import defaultdict


def get_prob(sent_list):
    sent_list = [sent.strip() for sent in sent_list if len(sent.strip()) in [5, 7]]
    word_count_dict = defaultdict(int)
    total = 0
    for text in sent_list:
        if len(text) == 5:
            for word in [text[:2], text[2:4], text[4]]:
                word_count_dict[word] += 0.5
            for word in [text[:2], text[2], text[3:5]]:
                word_count_dict[word] += 0.5
            total += 3
        else:
            for word in [text[:2], text[2:4], text[4:6], text[6]]:
                word_count_dict[word] += 0.5
            for word in [text[:2], text[2:4], text[4], text[5:7]]:
                word_count_dict[word] += 0.5
            total += 4
    word_prob_dict = {word: count/total for word, count in word_count_dict.items()}
    return word_prob_dict


def get_pmi(word, word_prob_dict, thr=1):  # 长度为2
    assert ValueError, f'{len(word) != 2}'
    pmi = word_prob_dict[word]/word_prob_dict.get(word[0], thr)/word_prob_dict.get(word[1], thr)
    return pmi


def template_pmi_select(sent, word_prob_dict):
    word1 = sent[-3:-1]
    word2 = sent[-2:]
    if len(sent) == 5:
        if get_pmi(word1, word_prob_dict) > get_pmi(word2, word_prob_dict):
            template = [2, 2, 1]
        else:
            template = [2, 1, 2]
    elif len(sent) == 7:
        if get_pmi(word1, word_prob_dict) > get_pmi(word2, word_prob_dict):
            template = [2, 2, 2, 1]
        else:
            template = [2, 2, 1, 2]
    else:
        assert ValueError, f'{len(sent) not in [5, 7]}'

    seg = []
    ix = 0
    for l in template:
        seg.append(sent[ix:ix+l])
        ix += l
    return seg


# 切分 sent_list
def template_pmi_select_seg(sent_list):
    word_prob_dict = get_prob(sent_list)
    seg_list = []
    for sent in sent_list:
        seg_list.append(template_pmi_select(sent, word_prob_dict))
    return seg_list


if __name__ == '__main__':

    inputfile = join(data_dir, 'modify_text57.txt')
    filedir = join(output_dir, '20221013', 'other_method')

    method = 'pmi_temp'
    sent_list = [d.strip() for d in read_txt_encoding(inputfile)]
    seg_list = template_pmi_select_seg(sent_list)

    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录
    integrate_result(method, seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)


