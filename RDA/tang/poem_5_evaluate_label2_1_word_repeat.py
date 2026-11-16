# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 13:14
# @Author  : Changzai Pan
# 用label1中的数据，测评分词方法的结果

import pandas as pd
from utils_seg import load_json, output_json
from RDA.tang.poem_0_config import *
from poem_5_utils import *
import numpy as np

def get_word_dict(poem_data, tau_l):
    from collections import Counter
    word_dict = dict(Counter([sent[i: j + 1] for data in poem_data
                              for sent in data['sent_list']
                              for i in range(len(sent))
                              for j in range(i, min(i + tau_l, len(sent)))
                              ]))
    return word_dict


def get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, eval_list):
    score_line = []
    print('total')
    tech_dict = get_tech_dict(sample_sent_seg_result_list, word_dict)
    score_dict1 = get_dict_result(word_list, tech_dict, word_dict, freq_thr, 2)
    score_dict2 = get_dict_result_len(word_list, tech_dict, word_dict, freq_thr, 2)
    score_dict3 = get_dict_result_len(word_list, tech_dict, word_dict, freq_thr, 3)
    score_line.extend([score_dict[eval]
                       for score_dict in [score_dict1, score_dict2, score_dict3]
                       for eval in eval_list])
    return score_line


def get_seg_result_with_type(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
                             sample_sent_type_list, type_list, eval_list=['f1', 'acc'], mode=1):
    score_line = []
    print('total')
    score_dict = get_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode)
    score_line.extend([score_dict[eval] for eval in eval_list])
    for type in type_list:
        print(type)
        sample_sent_idx_list_tmp = [idx for idx, type_tmp in zip(sample_sent_idx_list, sample_sent_type_list) if type_tmp == type]
        sample_sent_seg_result_list_tmp = [seg for seg, type_tmp in zip(sample_sent_seg_result_list, sample_sent_type_list) if type_tmp == type]
        score_dict_tmp = get_result(seg_text_dir, sample_sent_idx_list_tmp, sample_sent_seg_result_list_tmp, mode)
        score_tmp = [score_dict_tmp[eval] for eval in eval_list if eval in score_dict_tmp]
        score_line.extend(score_tmp)
    return score_line


def output_final_compare_result(type_list, eval_list, sample_sent_idx_list, sample_sent_seg_result_list,
                                sample_sent_type_list, word_dict, freq_thr):

    matrix = []
    for method in ['ltp', 'stanfordnlp', 'thulac', 'pkuseg', 'jieba']:  #, 'pmi_temp']:
        print(method)
        seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
        word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
        score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, ['word_rec', 'voc_size'])
        word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
        score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, ['word_rec', 'voc_size'])
        # score2 = [''] * 6
        score_line = get_seg_result_with_type(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
                                             sample_sent_type_list, type_list, eval_list)
        print(method, score_line[0])
        matrix.append([method] + score1 + score2 + score_line)

    # topwords
    result_dir = join(output_dir, '20221215', 'tw_single', 'topwords_seg_inf_inf')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, ['word_rec', 'voc_size'])
    word_list = get_dictionary_from_tw_dict_result(result_dir)
    score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, ['word_rec', 'voc_size'])
    score_line = get_seg_result_with_type(join(result_dir, 'segmented_text.txt'), sample_sent_idx_list, sample_sent_seg_result_list,
                                              sample_sent_type_list, type_list, eval_list)
    print('TopWORDS', score_line[0])
    matrix.append(['TopWORDS'] + score1 + score2 + score_line)

    # topwords-seg
    result_dir = join(output_dir, '20221215', 'tw_seg', 'topwords_seg_1e-05_0.1')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    word_list = get_dictionary_from_tw_dict_result(result_dir)
    score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    score_line = get_seg_result_with_type(join(result_dir, 'segmented_text.txt'), sample_sent_idx_list,
                                          sample_sent_seg_result_list,
                                          sample_sent_type_list, type_list, eval_list)
    print('TopWORDS-Seg', score_line[0])
    matrix.append(['TopWORDS-Seg'] + score1 + score2 + score_line)

    for method in ['rand_temp', 'pmi_temp']:
        seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
        word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
        score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                        ['word_rec', 'voc_size'])
        word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
        score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, ['word_rec', 'voc_size'])
        # score2 = [''] * 6
        score_line = get_seg_result_with_type(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
                                              sample_sent_type_list, type_list, eval_list)
        print(method, score_line[0])
        matrix.append([method] + score1 + score2 + score_line)

    # topwords-poetry1
    result_dir = join(output_dir, '20221215', 'mix23_single', 'topwords_seg_0.5_0.001')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    # word_list = get_dictionary_from_tw_dict_result(result_dir, para_thr=1e-7, score_thr=1)
    word_list = get_dictionary_from_tw_combine_result(result_dir, mode=1, para_thr=1e-7, level=0.05)
    score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    score_line = get_seg_result_with_type(join(result_dir, 'segmented_text.txt'), sample_sent_idx_list,
                                          sample_sent_seg_result_list,
                                          sample_sent_type_list, type_list, eval_list)
    print('TopWORDS-Poetry1', score_line[0])
    matrix.append(['TopWORDS-Poetry 0.001'] + score1 + score2 + score_line)

    # topwords-poetry2
    result_dir = join(output_dir, '20221215', 'mix23_double', 'topwords_seg_0.5_0.001')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=2)
    score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    #word_list = get_dictionary_from_tw_dict_result(result_dir, para_thr=1e-7, score_thr=1)
    word_list = get_dictionary_from_tw_combine_result(result_dir, mode=1, para_thr=1e-7, level=0.05)
    score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr,
                                    ['word_rec', 'voc_size'])
    score_line = get_seg_result_with_type(join(result_dir, 'segmented_text.txt'), sample_sent_idx_list,
                                          sample_sent_seg_result_list,
                                          sample_sent_type_list, type_list, eval_list, mode=2)
    print('TopWORDS-Poetry2', score_line[0])
    matrix.append(['TopWORDS-Poetry2 0.001'] + score1 + score2 + score_line)

    return matrix


def bootstrap(sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list, sample_sent_type_list, p=0.8):

    poem_type_to_id = defaultdict(list)
    for poem_id, poem_type in zip(sample_poem_idx_list, sample_sent_type_list):
        if poem_id not in poem_type_to_id[poem_type]:
            poem_type_to_id[poem_type].append(poem_id)

    poem_type_to_sample_id = defaultdict(list)
    sample_poem_id_list = []
    for poem_type, poem_id_list in poem_type_to_id.items():
        sample_size = int(p * len(poem_id_list))
        poem_type_to_sample_id[poem_type] = list(np.random.choice(poem_id_list, sample_size, replace=False))
        sample_poem_id_list.extend(poem_type_to_sample_id[poem_type])

    sample_sent_idx_list_new = []
    sample_sent_seg_result_list_new = []
    sample_sent_type_list_new = []
    for ix, poem_id in enumerate(sample_poem_idx_list):
        if poem_id in sample_poem_id_list:
            sample_sent_idx_list_new.append(sample_sent_idx_list[ix])
            sample_sent_seg_result_list_new.append(sample_sent_seg_result_list[ix])
            sample_sent_type_list_new.append(sample_sent_type_list[ix])

    return sample_sent_idx_list_new, sample_sent_seg_result_list_new, sample_sent_type_list_new



if __name__ == '__main__':

    # 将模型结果和标注数据对比，输出result.xlsx结果
    # 读取模型文件，输出template的比例

    df_sample_label = pd.read_excel(join(data_dir, 'label2', 'sample_data1000.xlsx'))
    poem_data = load_json(join(data_dir, 'poem_data_57.json'))

    sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list, sample_sent_type_list = get_seg_idx_and_result_with_type(
        df_sample_label, poem_data)
    print(check_idx_and_result(sample_sent_idx_list, sample_sent_seg_result_list))

    word_dict = get_word_dict(poem_data, tau_l=7)

    ###########################################
    # 抽样n次
    type_list = [(5, 4), (5, 8), (7, 4), (7, 8)]
    eval_list = ['f1']
    freq = 2
    col_precent = [1, 3, 5, 7, 9, 11, 13, 14, 15, 16, 17]
    col_num = [2, 4, 6, 8, 10, 12]
    matrix_samples = [[[] for _ in range(18)] for _ in range(11)]
    np.random.seed(1)
    for k in range(20):
        tmp = bootstrap(sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list, sample_sent_type_list)
        sample_sent_idx_list_new, sample_sent_seg_result_list_new, sample_sent_type_list_new = tmp
        matrix = output_final_compare_result(type_list, eval_list,
                                             sample_sent_idx_list_new, sample_sent_seg_result_list_new,
                                             sample_sent_type_list_new,
                                             word_dict, freq_thr=freq)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if j in col_precent:
                    matrix_samples[i][j].append(float(matrix[i][j][:-1]))
                elif j in col_num:
                    matrix_samples[i][j].append(int(matrix[i][j]))
                else:
                    matrix_samples[i][j] = matrix[i][j]

    output_json(matrix_samples, join(output_dir, '20221215', f'matrix_samples.json'))

    matrix_samples = load_json(join(output_dir, '20221215', 'matrix_samples.json'))
    # 组合
    matrix_dist = [[[] for _ in range(18)] for _ in range(11)]
    for i in range(11):
        for j in range(18):
            if j in col_precent:
                mean = np.mean(matrix_samples[i][j])
                std = np.std(matrix_samples[i][j])
                matrix_dist[i][j] = str(round(mean, 2)) + '%' + ' ± ' + str(round(std, 2)) + '%'
            elif j in col_num:
                mean = np.mean(matrix_samples[i][j])
                std = np.std(matrix_samples[i][j])
                matrix_dist[i][j] = str(int(mean)) + ' ± ' + str(int(std))
            else:
                matrix_dist[i][j] = matrix_samples[i][j]

    micolumns = pd.MultiIndex.from_tuples(
        [('method', 'method')] +
        [(type, eval + ' ' + cond)
         for type in ['seg_1000', 'seg_whole']
         for cond in ['>=2', '2', '3']
         for eval in ['word_rec', 'voc_size']] +
        [(type, eval) for type in ['total'] + type_list for eval in eval_list],
        names=['data', 'eval']
    )
    df_result = pd.DataFrame(matrix_dist, columns=micolumns)
    df_result.to_excel(join(output_dir, '20221215', f'summary_result_{freq}_distribtion1.xlsx'))

    import matplotlib.pyplot as plt
    from scipy import stats
    def ttest(col_ix, row_ix1, row_ix2):
        pvalue = stats.mstats.ttest_rel(matrix_samples[row_ix1][col_ix], matrix_samples[row_ix2][col_ix]).pvalue
        # plt.subplot(1, 2, 1)
        # plt.hist(matrix_samples[row_ix1][col_ix])
        # plt.subplot(1, 2, 2)
        # plt.hist(matrix_samples[row_ix2][col_ix])
        dis = np.mean(matrix_samples[row_ix1][col_ix]) - np.mean(matrix_samples[row_ix2][col_ix])
        return round(dis, 2), '{:.3e}'.format(pvalue)

    print('R_s(2,2)', ttest(7, 10, 7))
    print('R_s(2,3)', ttest(11, 10, 5))
    print('R_d(2,2)', ttest(1, 10, 8))
    print('R_d(2,3)', ttest(5, 10, 5))
    print('F_s(P_1000)', ttest(13, 10, 8))
    print('F_s(45)', ttest(14, 10, 8))
    print('F_s(85)', ttest(15, 10, 8))
    print('F_s(47)', ttest(16, 10, 8))
    print('F_s(87)', ttest(17, 10, 8))
