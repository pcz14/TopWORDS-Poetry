# -*- coding: utf-8 -*-
import numpy as np

from RDA.tang.poem_0_config import *
from utils_seg import read_txt_encoding
from TopwordsSegMix2.TopwordsSeg import get_sent_prior_from_poem, get_sent_prior_from_poem2


if __name__ == '__main__':

    # 建立 sent_prior.pkl
    get_sent_prior_from_poem(input_file=join(data_dir, 'modify_text57.txt'),
                             output_dir=join(data_dir, 'sent_prior.pkl'),
                             template={5: [[2, 2, 1], [2, 1, 2]], 7: [[2, 2, 2, 1], [2, 2, 1, 2]]})

    get_sent_prior_from_poem(input_file=join(data_dir, 'modify_text57.txt'),
                             output_dir=join(data_dir, 'sent_prior_3.pkl'),
                             template={5: [[2, 2, 1], [2, 1, 2], [2, 1, 1, 1]],
                                       7: [[2, 2, 2, 1], [2, 2, 1, 2], [2, 2, 1, 1, 1]]})

    get_sent_prior_from_poem2(input_file=join(data_dir, 'modify_text572.txt'),
                             output_dir=join(data_dir, 'sent_prior2.pkl'),
                              template={10: [[2, 1, 2, 2, 1, 2], [2, 2, 1, 2, 1, 2],
                                             [2, 1, 2, 2, 2, 1], [2, 2, 1, 2, 2, 1]],
                                        14: [[2, 2, 1, 2, 2, 2, 1, 2], [2, 2, 2, 1, 2, 2, 1, 2],
                                             [2, 2, 1, 2, 2, 2, 2, 1], [2, 2, 2, 1, 2, 2, 2, 1]]})

    get_sent_prior_from_poem(input_file=join(data_dir, 'modify_text57.txt'),
                             output_dir=join(data_dir, 'sent_prior_comp23.pkl'),
                             template={5: [[2, 2, 1], [2, 1, 2], [2, 3]],
                                       7: [[2, 2, 2, 1], [2, 2, 1, 2], [2, 2, 3]]})

    # input_file = join(data_dir, 'modify_text572.txt')
    # tmp = read_txt_encoding(input_file)
    import pickle
    with open(join(data_dir, 'sent_prior_comp23.pkl'), 'rb') as f:
        sent_prior_dict = pickle.loads(f.read())


import numpy as np
def corr(mat):
    a, b, c, d = mat
    frac1 = a * d - b * c
    frac2 = np.sqrt((a + b) * (a + c) * (c + d) * (b + d))
    return frac1/frac2