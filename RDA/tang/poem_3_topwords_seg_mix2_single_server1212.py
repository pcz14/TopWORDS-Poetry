# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:55
# @Author  : Changzai Pan
# @File    : wiki_topwords_seg.py

import os
from os.path import join
import numpy as np
import sys
print(os.getcwd())
sys.path.append(os.getcwd())


if __name__ == '__main__':

    work_dir = '/data/run01/scv7387/project/Poetry_Seg'

    data_dir = join(work_dir, 'RDA', 'tang', 'data')
    output_dir = join(work_dir, 'RDA', 'tang', 'output', '20221214')

    inputfile = join(data_dir, 'modify_text57.txt')
    filedir = join(output_dir, 'mix23_single')
    sent_prior_dir = join(data_dir, 'sent_prior.pkl')

    from TopwordsSegMix2.TopwordsSeg import topwords_seg

    # 保护 word_candidate， 保护prior_words_cbdb
    # 保护 word_candidate 2-3中的2，后面的3不保护
    # 前面和后面的pseudo count不一样

    for kappa_d in [0.5]:
        for kappa_s in [0.001, 0.5, 0.1, 0.05, 0.02, 0.01, 0.0001, 0.00001]:
            topwords_seg(inputfile=inputfile, filedir=filedir,
                         output_dir=None, method_name=None,
                         model_path=None,
                         tau_l=3, tau_f=2,
                         seg_thr=0.5,
                         kappa_d=kappa_d,  # parameter estimation
                         kappa_s=kappa_s,  # segmetation
                         comp_value={5: np.array([0.5, 0.5]), 7: np.array([0.5, 0.5])},
                         prior_words_dir=None,
                         protected_prior_words_dir=join(data_dir, 'word_candidate2.txt'),
                         delete_words_dir=None,
                         sent_prior_dir=sent_prior_dir,
                         num_of_processes=10)
