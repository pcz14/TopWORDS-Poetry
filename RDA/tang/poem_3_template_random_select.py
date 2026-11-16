# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:49
# @Author  : Changzai Pan
# @File    : wiki_topwords.py

from RDA.tang.poem_0_config import *
import random
from utils_seg import read_txt_encoding, check_result_dir, integrate_result

def template_random_select(sent):
    idx = random.random() > 0.5
    if len(sent) == 5:
        if idx:
            template = [2, 2, 1]
        else:
            template = [2, 1, 2]
    elif len(sent) == 7:
        if idx:
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
def template_random_select_seg(sent_list):
    return [template_random_select(sent) for sent in sent_list]


if __name__ == '__main__':

    inputfile = join(data_dir, 'modify_text57.txt')
    filedir = join(output_dir, '20221013', 'other_method')

    method = 'rand_temp'
    sent_list = [d.strip() for d in read_txt_encoding(inputfile)]
    seg_list = template_random_select_seg(sent_list)

    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录
    integrate_result(method, seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)


