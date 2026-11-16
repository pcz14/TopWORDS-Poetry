# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 13:14
# @Author  : Changzai Pan
# 用label1中的数据，测评分词方法的结果

import pandas as pd
from utils_seg import load_json, is_chinese_character
from RDA.tang.poem_0_config import *
from poem_5_utils import *


def get_word_dict(poem_data, tau_l):
    from collections import Counter
    word_dict = dict(Counter([sent[i: j + 1] for data in poem_data
                              for sent in data['sent_list']
                              for i in range(len(sent))
                              for j in range(i, min(i + tau_l, len(sent)))
                              ]))
    return word_dict


def get_entity_recall_result(word_list, cate_to_entity):
    cate_to_score = {}
    for cate, words in cate_to_entity.items():
        cate_to_score[cate] = len(set(words) & set(word_list))/len(words)
    return cate_to_score


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


def output_poetry_compare_result(eval_list, sample_sent_idx_list, sample_sent_seg_result_list,
                                 word_dict, freq_thr,
                                 method_list, method_kappa_dict, method_mode_dict):

    matrix = []
    for method in method_list:
        for kappa_d in method_kappa_dict[method][0]:
            for kappa_s in method_kappa_dict[method][1]:
                mode = method_mode_dict[method]
                seg_dir = join(output_dir, '20221215', method)
                # 1000句的分词结果
                word_list = get_dictionary_from_tw_seg_result(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}'),
                                                              sample_sent_idx_list, sample_sent_seg_result_list, mode)
                score1 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, eval_list)
                # 全文的分词结果
                word_list = get_dictionary_from_tw_seg_result_whole(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}'), mode)
                score2 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, eval_list)
                # 字典结果
                word_list = get_dictionary_from_tw_dict_result(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}'))
                score3 = get_word_recall_result(sample_sent_seg_result_list, word_list, word_dict, freq_thr, eval_list)

                matrix.append([method, mode, kappa_d, kappa_s] + score1 + score2 + score3)

    micolumns = pd.MultiIndex.from_tuples(
        [('method', 'method'), ('mode', 'mode'), ('kappa_d', 'kappa_d'), ('kappa_s', 'kappa_s')] +
        [(type, eval + ' ' + cond) for type in ['seg_1000', 'seg_whole', 'dict'] for cond in ['>=2', '2', '3'] for eval in eval_list],
        names=['type', 'eval'])
    df_poetry = pd.DataFrame(matrix, columns=micolumns)
    df_poetry.to_excel(join(output_dir, '20221215', f'result_compare_dict_{freq_thr}.xlsx'))


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

    micolumns = pd.MultiIndex.from_tuples(
        [('method', 'method')] +
        [(type, eval + ' ' + cond)
         for type in ['seg_1000', 'seg_whole']
         for cond in ['>=2', '2', '3']
         for eval in ['word_rec', 'voc_size']] +
        [(type, eval) for type in ['total'] + type_list for eval in eval_list],
        names=['data', 'eval']
    )
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

    df_result = pd.DataFrame(matrix, columns=micolumns)
    df_result.to_excel(join(output_dir, '20221215', f'summary_result_{freq_thr}.xlsx'))



if __name__ == '__main__':

    # 评估模型对于实体的recall
    # 1000文本上的所有三种类别实体，并计算三种实体的recall

    ###############################################
    # 先导出所有测试诗歌上切分出的词汇
    df_sample_label = pd.read_excel(join(data_dir, 'label2', 'sample_data1000.xlsx'))
    poem_data = load_json(join(data_dir, 'poem_data_57.json'))
    sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list, sample_sent_type_list = get_seg_idx_and_result_with_type(
        df_sample_label, poem_data)
    print(check_idx_and_result(sample_sent_idx_list, sample_sent_seg_result_list))
    word_test_list = list(set([w for s in sample_sent_seg_result_list for w in s.split('/')]))

    df_word = pd.DataFrame({'word': word_test_list, 'len':[len(word) for word in word_test_list]})  # 将要导出的excel
    word_context_dict = defaultdict(list)
    for s in sample_sent_seg_result_list:
        for w in s.split('/'):
            word_context_dict[w].append(s)

    # 整理已有实体集合:
    # 1. 标注过的词汇， 得到 cate_to_entity
    df_entity = pd.read_excel(join(output_dir, '20221215', 'label_entity.xlsx'))
    tmp_dict = {1:'name', 2:'address', 3:'allusion'}
    cate_to_entity = {}
    for ix, name in tmp_dict.items():
        cate_to_entity[name] = list(df_entity[df_entity.label == ix]['word'])
    for cate, words in cate_to_entity.items():
        print(cate, len(words))

    # 2. 古代大辞典和六专名， 得到 cate_to_entity_out
    from opencc import OpenCC
    cc = OpenCC('s2t')
    cate_to_entity_out = {}
    cate_to_entity_out['name'] = read_txt_encoding(join(data_dir, '中国历代人名大辞典（底本）.txt'))
    cate_to_entity_out['address'] = read_txt_encoding(join(data_dir, '中国古典诗词地名辞典.txt'))
    cate_to_entity_out['allusion'] = read_txt_encoding(join(data_dir, '全唐诗典故辞典.txt'))
    for cate, words in cate_to_entity_out.items():
        tmp = []
        for w in words:
            t = ''
            for s in w:
                if is_chinese_character(s):
                    t += s
                else:
                    break
            if t != '' and len(t) <= 4:
                tmp.append(cc.convert(t))
        cate_to_entity_out[cate] = tmp

    df_cbdb = pd.read_excel(join(data_dir, '六专名18.3.13.xlsx'))
    cate_to_entity_out['name'].extend([cc.convert(w) for w in df_cbdb[df_cbdb.type == '人名']['word'] if 1 < len(w) < 5])
    cate_to_entity_out['address'].extend([cc.convert(w) for w in df_cbdb[df_cbdb.type == '地名']['word'] if 1 < len(w) < 5])
    for cate, words in cate_to_entity_out.items():
        cate_to_entity_out[cate] = list(set(words))
        print(cate, len(words))

    # 对已有测试诗歌上的词汇进行筛选
    df_word['freq'] = [len(word_context_dict[word]) for word in word_test_list]
    df_word['context'] = [' \n'.join(word_context_dict[word]) for word in word_test_list]
    df_word['name'] = [1 if w in cate_to_entity['name'] or w in cate_to_entity_out['name'] else 0 for w in word_test_list]
    df_word['address'] = [1 if w in cate_to_entity['address'] or w in cate_to_entity_out['address'] else 0 for w in word_test_list]
    df_word['allusion'] = [1 if w in cate_to_entity['allusion'] or w in cate_to_entity_out['allusion'] else 0 for w in word_test_list]
    df_word['entity'] = df_word['name'] + df_word['address'] + df_word['allusion']
    df_word_new = df_word[(df_word.len != 1) & (df_word['entity'] == 0)]
    print(len(df_word_new))
    df_word_new.to_excel(join(output_dir, '20221215', 'test_data_words_label.xlsx'))
    df_word.to_excel(join(output_dir, '20221215', 'test_data_words.xlsx'))

    ###############################################
    # 人工标注


    ###############################################
    # 获取标注的实体列表
    df_test_label = pd.read_excel(join(output_dir, '20221215', '标注材料-人工校对后.xlsx'))
    cate_to_entity = {}
    cate_to_entity['name'] = list(df_test_label[df_test_label['name'] == 1].word)
    cate_to_entity['address'] = list(df_test_label[df_test_label['address'] == 1].word)
    cate_to_entity['allusion'] = list(df_test_label[df_test_label['allusion'] == 1].word)
    for cate, words in cate_to_entity.items():
        print(cate, len(words))

    df_test_data = pd.read_excel(join(output_dir, '20221215', 'test_data_words.xlsx'))
    cate_to_entity['name'].extend(list(df_test_data[df_test_data['name'] == 1].word))
    cate_to_entity['address'].extend(list(df_test_data[df_test_data['address'] == 1].word))
    cate_to_entity['allusion'].extend(list(df_test_data[df_test_data['allusion'] == 1].word))
    for cate, words in cate_to_entity.items():
        print(cate, len(words))

    ###########################################
    # 整理所有方法的结果
    # 有监督方法
    type_list = [(5, 4), (5, 8), (7, 4), (7, 8)]
    eval_list = ['f1']
    freq_thr = 2
    cate_list = ['name', 'address', 'allusion']

    micolumns = pd.MultiIndex.from_tuples(
        [('method', 'method')] +
        [(type, cate)
         for type in ['seg_1000', 'seg_whole']
         for cate in cate_list],
        names=['data', 'eval']
    )
    matrix = []
    for method in ['ltp', 'stanfordnlp', 'thulac', 'pkuseg', 'jieba']:  # , 'pmi_temp']:
        print(method)
        seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
        word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
        cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
        word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
        cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
        matrix.append([method] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    # topwords
    result_dir = join(output_dir, '20221215', 'tw_single', 'topwords_seg_inf_inf')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    word_list = get_dictionary_from_tw_dict_result(result_dir)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['TopWORDS'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    # topwords-seg
    result_dir = join(output_dir, '20221215', 'tw_seg', 'topwords_seg_1e-05_0.1')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    word_list = get_dictionary_from_tw_dict_result(result_dir)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['TopWORDS-Seg'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    for method in ['rand_temp', 'pmi_temp']:
        seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
        word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
        cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
        word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
        cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
        matrix.append([method] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    # topwords-poetry1
    result_dir = join(output_dir, '20221215', 'mix23_single', 'topwords_seg_0.5_0.001')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=1)
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    # word_list = get_dictionary_from_tw_dict_result(result_dir, para_thr=1e-7, score_thr=1)
    word_list = get_dictionary_from_tw_combine_result(result_dir, mode=1, para_thr=1e-7, level=0.05)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['TopWORDS-Poetry 0.001'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    # topwords-poetry2
    result_dir = join(output_dir, '20221215', 'mix23_double', 'topwords_seg_0.5_0.001')
    word_list = get_dictionary_from_tw_seg_result(result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode=2)
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    # word_list = get_dictionary_from_tw_dict_result(result_dir, para_thr=1e-7, score_thr=1)
    word_list = get_dictionary_from_tw_combine_result(result_dir, mode=1, para_thr=1e-7, level=0.05)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['TopWORDS-Poetry2 0.001'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    # chatgpt
    method = 'chatgpt'
    seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
    sample_sent_idx_list = list(range(6000))
    word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
    word_list = list(set(word_list))
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['ChatGPT'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    method = 'chatgpt4new'
    seg_text_dir = join(output_dir, '20221215', 'other_method', 'seg_result', f'{method}.txt')
    sample_sent_idx_list = list(range(6000))
    word_list = get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list)
    word_list = list(set(word_list))
    cate_to_score = get_entity_recall_result(word_list, cate_to_entity)
    word_list = get_dictionary_from_seg_result_whole(seg_text_dir)
    cate_to_score2 = get_entity_recall_result(word_list, cate_to_entity)
    matrix.append(['GPT4o'] + [cate_to_score[cate] for cate in cate_list] + [cate_to_score2[cate] for cate in cate_list])

    df_result = pd.DataFrame(matrix, columns=micolumns)
    df_result.to_excel(join(output_dir, '20221215', f'summary_result_{freq_thr}_entitynew.xlsx'))






