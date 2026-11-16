# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:55
# @Author  : Changzai Pan
# @File    : wiki_topwords_seg.py

import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from RDA.tang.poem_0_config import *
from TopwordsSeg.TopwordsSeg import topwords_seg, get_sent_prior_from_methods
from utils_seg import load_json


if __name__ == '__main__':

    inputfile = join(data_dir, 'modify_text57.txt')
    sent_prior_dir = join(data_dir, 'sent_prior.pkl')
    filedir = join(output_dir, '20221201', 'tw_seg')

    if not os.path.exists(filedir):
        os.mkdir(filedir)

    # get_sent_prior_from_methods(
    #     seg_01_result_dir=join(output_dir, '20221013', 'other_method', 'seg_01_result'),
    #     output_dir=join(output_dir, '20221201', 'tw_seg', 'sent_prior_pkuseg.pkl'),
    #     method_list=['pkuseg'])

    for kappa_s in [0.1, 0.01, 0.001, 0.0001, 1e-5]:
        topwords_seg(inputfile=inputfile, filedir=filedir,
                     output_dir=None, method_name=None,
                     model_path=None,
                     tau_l=3, tau_f=2,
                     seg_thr=0.5,
                     kappa_d=float('inf'),  # parameter estimation
                     kappa_s=float('inf'),  # segmetation
                     prior_words_dir=join(data_dir, 'word_candidate.txt'),
                     protected_prior_words_dir=join(data_dir, 'prior_words_tang.txt'),
                     sent_prior_dir=None,
                     num_of_processes=4)