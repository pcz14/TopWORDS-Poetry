# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 13:10
# @Author  : Changzai Pan
# @File    : wiki_3_othermethod.py


from RDA.tang.poem_0_config import *
from other_method import segment_other_method, segment_combine_method

if __name__ == '__main__':

    inputfile = join(data_dir, 'modify_text57.txt')
    filedir = join(output_dir, '20221013', 'other_method')
    segment_other_method(inputfile, filedir, methods=['pkuseg', 'ltp'],
                         user_dict_dir=None, num_of_processes=1, log_file=join(filedir, 'log.txt'),
                         modeldir=r'D:\pycharm\PycharmProjects\TopWORDS-Seg\models\stanford-corenlp-full-2018-10-05')

    segment_combine_method(inputfile, filedir, methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                           log_file=join(filedir, 'log.txt'))
