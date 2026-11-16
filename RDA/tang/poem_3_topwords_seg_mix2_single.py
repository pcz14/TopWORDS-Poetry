# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:55
# @Author  : Changzai Pan
# @File    : wiki_topwords_seg.py

import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from RDA.tang.poem_0_config import *
from TopwordsSegMix2.TopwordsSeg import topwords_seg
import numpy as np

if __name__ == '__main__':

    kappa_d = 0.001

    inputfile = join(data_dir, 'modify_text57.txt')
    sent_prior_dir = join(data_dir, 'sent_prior.pkl')
    filedir = join(output_dir, '20221201', 'mix23_single')

    for kappa_s in [0.1, 0.01, 0.001, 0.0001, 1e-5]:
        topwords_seg(inputfile=inputfile, filedir=filedir,
                    output_dir=None, method_name=None,
                    model_path=None,
                    tau_l=3, tau_f=2,
                    seg_thr=0.5,
                    kappa_d=kappa_d,  # parameter estimation
                    kappa_s=kappa_s,  # segmetation
                    comp_value = {5: np.array([0.5, 0.5]), 7: np.array([0.5, 0.5])},
                    prior_words_dir=join(data_dir, 'prior_words_tang.txt'),
                    protected_prior_words_dir=join(data_dir, 'word_candidate.txt'),
                    # prior_words_dir=join(data_dir, 'word_candidate.txt'),
                    # protected_prior_words_dir=join(data_dir, 'prior_words_cbdb.txt'),
                    delete_words_dir=None,
                    sent_prior_dir=sent_prior_dir,
                    num_of_processes=8)
        
