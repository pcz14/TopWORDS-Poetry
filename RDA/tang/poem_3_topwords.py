# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:49
# @Author  : Changzai Pan
# @File    : wiki_topwords.py

import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from RDA.tang.poem_0_config import *
from TopwordsSeg.TopwordsSeg import topwords_seg
import argparse
import numpy as np

if __name__ == '__main__':

    inputfile = join(data_dir, 'modify_text57.txt')
    filedir = join(output_dir, '20221201', 'tw')
    sent_prior_dir = join(data_dir, 'sent_prior.pkl')

    topwords_seg(inputfile=inputfile, filedir=filedir,
                    output_dir=None, method_name=None,
                    model_path=None,
                    tau_l=3, tau_f=2,
                    seg_thr=0.5,
                    kappa_d=float('inf'),  # parameter estimation
                    kappa_s=float('inf'),  # segmetation
                    prior_words_dir=join(data_dir, 'prior_words_tang.txt'),
                    protected_prior_words_dir=join(data_dir, 'word_candidate.txt'),
                    delete_words_dir=None,
                    sent_prior_dir=None,
                    num_of_processes=4)