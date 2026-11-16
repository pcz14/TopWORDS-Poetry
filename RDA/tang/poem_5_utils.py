# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 13:14
# @Author  : Changzai Pan
# @File    : wiki_6_evaluate.py


from utils_seg import read_txt_encoding, seg_score
from RDA.tang.poem_0_config import *
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import chi2

# 根据标注的部分诗歌，寻找其在整体全唐诗中的行位置和诗歌idx
def get_seg_idx_and_result(df_sample_label, poem_data):
    true_poem_data = [[sent for col, sent in line.items()
                       if col != '编号' and not pd.isna(sent) and sent.strip() != '']
                      for ix, line in df_sample_label.iterrows()]
    true_poem_data = [line for line in true_poem_data if len(line) in [4, 8]]

    # 找到句子的真实标签，以及对应的文本行数

    text2ix = {}
    text2id = {}
    line_ix = 0
    for data in poem_data:
        text = ''.join(data['sent_list'])
        l = len(data['sent_list'])
        ix_list = list(range(line_ix, line_ix + l))
        text2ix[text] = ix_list
        text2id[text] = [data['id'] for _ in ix_list]
        line_ix += l

    sample_sent_idx_list = []
    sample_sent_seg_result_list = []
    sample_poem_idx_list = []
    k = 0
    for data in true_poem_data:
        text = ''.join([''.join(sent.split('/')) for sent in data])
        if text in text2ix:
            k += 1
            sample_sent_idx_list.extend(text2ix[text])
            sample_sent_seg_result_list.extend(data)
            sample_poem_idx_list.extend(text2id[text])
    print('有效的数量', k)
    return sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list


def get_seg_idx_and_result_with_type(df_sample_label, poem_data):

    sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list = get_seg_idx_and_result(df_sample_label, poem_data)

    poem_data_dict = {data['id']: data for data in poem_data}
    sample_sent_type_list = [(poem_data_dict[poem_idx]['yan'], poem_data_dict[poem_idx]['ju'])
                             for poem_idx in sample_poem_idx_list]

    return sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list, sample_sent_type_list


def check_idx_and_result(sample_sent_idx_list, sample_sent_seg_result_list):
    text_string = read_txt_encoding(join(data_dir, 'modify_text57.txt'))
    for idx, seg in zip(sample_sent_idx_list, sample_sent_seg_result_list):
        if text_string[idx].strip() != ''.join(seg.split('/')):
            return False
    return True


#################################################################
# EDA
# 多少个5言和7言的，4句和8句的
def get_57_count(sample_poem_idx_list, poem_data):
    count = defaultdict(int)
    data_dict= {data['id']:data for data in poem_data}
    for poem_idx in set(sample_poem_idx_list):
        yan = data_dict[poem_idx]['yan']
        ju = data_dict[poem_idx]['ju']
        count[f'{yan}{ju}'] += 1
    print(count)
    return count

# 长于2的词汇
def get_long_word(sample_sent_seg_result_list):
    long_word_list = []
    tmp = [true_seg.split('/') for true_seg in sample_sent_seg_result_list]
    for t in tmp:
        for word in t:
            if len(word) >= 3:
                print(t, word)
                long_word_list.append(word)
    return long_word_list

# 枚举各种分词模式以及数量
def pattern01_to_seg(pattern01):  # (0, 1, 0, 1, 1) -> 2-2-1
    len_tmp = 1
    result = []
    for b in pattern01:
        if b == 1:
            result.append(len_tmp)
            len_tmp = 1
        else:
            len_tmp += 1
    return '-'.join([str(l) for l in result])


def get_pattern(sample_sent_seg_result_list, len_list=[5, 7]):
    true_01_list = [tuple(seg_to_01(true_seg, sep='/')) for true_seg in sample_sent_seg_result_list]
    from collections import Counter
    c = dict(Counter(true_01_list).most_common())
    sum_dict = {l: sum([count for pattern, count in c.items() if len(pattern) == l]) for l in len_list}
    df_true_pattern = pd.DataFrame({'pattern01': [pattern for pattern, count in c.items()],
                                    'pattern': [pattern01_to_seg(pattern) for pattern, count in c.items()],
                                    'len': [len(pattern) for pattern, count in c.items()],
                                    'count': [count for pattern, count in c.items()],
                                    'precent': [count/sum_dict[len(pattern)] for pattern, count in c.items()]})
    return df_true_pattern, true_01_list

# 上下句的相关性
def get_correlation12(true_01_list):
    count = 0
    for ix, tmp in enumerate(true_01_list):
        if ix % 2 == 0:
            if tmp == true_01_list[ix+1]:
                count += 1
    print(count/len(true_01_list)*2)
    return count/len(true_01_list)*2


#################################################################
# 评估分词
# 如果是两行为一个单位的，要切开
def cut_double_line(seg_data):
    seg_data_new = []
    for seg in seg_data:
        l = len(''.join(seg.split(' ')))
        assert l % 2 == 0, 'not double line'
        l0 = int(l/2)
        seg_list = [[], []]
        pos_ix = 0
        ix = 0 # 前半段还是后半段
        for s in seg.split(' '):
            seg_list[ix].append(s)
            pos_ix += len(s)
            if pos_ix == l0:
                ix = 1
        seg_data_new.extend([' '.join(tmp) for tmp in seg_list])
    return seg_data_new


def seg_to_01(seg, sep=' '):
    word_list = seg.split(sep)
    seg_01_list = []
    for word in word_list:
        seg_01_list.extend([0] * (len(word) - 1) + [1])
    return seg_01_list


def seg_score_poem(predict_seg_sent_list, true_seg_sent_list, alpha = 1/4):
    report = []
    for pred_seg, true_seg in zip(predict_seg_sent_list, true_seg_sent_list):
        predict_01_list = seg_to_01(pred_seg, sep=' ')
        true_01_list = seg_to_01(true_seg, sep='/')
        report.extend([(i, j) for i, j in zip(predict_01_list[:-1], true_01_list[:-1])])
    from collections import Counter
    c = Counter(report)
    acc = (c[(0, 0)] + c[(1, 1)] + c[(0, 1)]*alpha)/len(report)
    return acc


# 检查诗歌切分文本和金标准的文本是否一样
def check_seg_poem(pred_seg_data, sample_sent_seg_result_list):
    assert len(pred_seg_data) == len(sample_sent_seg_result_list), 'size not equal!'
    for i in range(len(sample_sent_seg_result_list)):
        true_tmp = ''.join(sample_sent_seg_result_list[i].split('/'))
        seg = ''.join(pred_seg_data[i].strip().split(' '))
        if seg != true_tmp:
            print(i, true_tmp, seg)
            break


def get_dictionary_from_seg_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list):  # type=adapt or seg_dict
    seg_data = read_txt_encoding(seg_text_dir)
    seg_data = [seg.strip() for seg in seg_data]
    pred_seg_data = [seg_data[ix] for ix in sample_sent_idx_list]
    check_seg_poem(pred_seg_data, sample_sent_seg_result_list)
    word_list = [word for seg in pred_seg_data for word in seg.split(' ')]
    return list(set(word_list))

def get_dictionary_from_seg_result_whole(seg_text_dir):  # type=adapt or seg_dict
    seg_data = read_txt_encoding(seg_text_dir)
    pred_seg_data = [seg.strip() for seg in seg_data]
    word_list = [word for seg in pred_seg_data for word in seg.split(' ')]
    return list(set(word_list))

def get_dictionary_from_tw_seg_result(seg_result_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode):
    seg_result_dir = join(seg_result_dir, 'segmented_text.txt')
    seg_data = read_txt_encoding(seg_result_dir)
    seg_data = [seg.strip() for seg in seg_data]
    if mode == 2:
        seg_data = cut_double_line(seg_data)
    pred_seg_data = [seg_data[ix] for ix in sample_sent_idx_list]
    check_seg_poem(pred_seg_data, sample_sent_seg_result_list)
    word_list = [word for seg in pred_seg_data for word in seg.split(' ')]
    return list(set(word_list))

def get_dictionary_from_tw_seg_result_whole(seg_result_dir, mode):
    seg_result_dir = join(seg_result_dir, 'segmented_text.txt')
    seg_data = read_txt_encoding(seg_result_dir)
    pred_seg_data = [seg.strip() for seg in seg_data]
    if mode == 2:
        pred_seg_data = cut_double_line(pred_seg_data)
    word_list = [word for seg in pred_seg_data for word in seg.split(' ')]
    return list(set(word_list))

def get_dictionary_from_tw_dict_result(seg_result_dir, para_thr=-1, score_thr=-1):
    df = pd.read_csv(join(seg_result_dir, 'word_dictionary.csv'))
    df = df[(df.word_score > score_thr)&(df.theta_value > para_thr)]
    return list(set(list(df['word'])))


def get_dictionary_from_tw_combine_result(result_dir, mode, para_thr=-1, level=0.05):
    # word from segmentation
    seg_data = read_txt_encoding(join(result_dir, 'segmented_text.txt'))
    pred_seg_data = [seg.strip() for seg in seg_data]
    if mode == 2:
        pred_seg_data = cut_double_line(pred_seg_data)
    word_list_seg = [word for seg in pred_seg_data for word in seg.split(' ')]
    print('seg', len(word_list_seg))
    # word from dictionary
    df = pd.read_csv(join(result_dir, 'word_dictionary.csv'))
    q = 1 - level / len(df)
    score_thr = chi2.ppf(q=q, df=1) / 2
    df = df[(df.word_score > score_thr)&(df.theta_value > para_thr)]
    word_list_dict = list(df['word'])
    print('word', len(word_list_dict))
    print('all', len(list(set(word_list_seg + word_list_dict))))
    return list(set(word_list_seg + word_list_dict))


def get_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
               mode=1):
    seg_data = read_txt_encoding(seg_text_dir)
    seg_data = [seg.strip() for seg in seg_data]
    if mode == 2:
        seg_data = cut_double_line(seg_data)
    # check:
    pred_seg_data = [seg_data[ix] for ix in sample_sent_idx_list]
    check_seg_poem(pred_seg_data, sample_sent_seg_result_list)

    score01 = seg_score_poem(pred_seg_data, sample_sent_seg_result_list, alpha=0)
    # score012 = seg_score_poem(pred_seg_data, sample_sent_seg_result_list, alpha=1 / 4)

    pred_01_list = [seg_to_01(s)[:-1] for s in [seg_data[ix] for ix in sample_sent_idx_list]]
    true_01_list = [seg_to_01(s, '/')[:-1] for s in sample_sent_seg_result_list]
    score = seg_score(pred_01_list, true_01_list)

    score_dict = {eval: "%.2f%%" % (score * 100) for eval, score in zip(['acc', 'prec', 'recall', 'f1'], [score01] + list(score))}
    return score_dict


def get_word_dict(poem_data, tau_l):
    from collections import Counter
    word_dict = dict(Counter([sent[i: j + 1] for data in poem_data
                              for sent in data['sent_list']
                              for i in range(len(sent))
                              for j in range(i, min(i + tau_l, len(sent)))
                              ]))
    return word_dict


def get_tech_dict(sample_sent_seg_result_list, word_dict):
    word_list = []
    for seg_result in sample_sent_seg_result_list:
        word_list.extend(seg_result.split('/'))
    tech_dict = {word: {'freq': word_dict[word], 'len': len(word)} for word in set(word_list)}
    return tech_dict


def get_dict_result(word_list, tech_dict, word_dict, freq_thr, len_thr):
    tech_list = set([word for word, attr in tech_dict.items()
                     if word_dict[word] >= freq_thr and len(word) >= len_thr])
    word_list = set(word_list)
    voc = set([word for word in word_list
                if word_dict.get(word, 0) >= freq_thr and len(word) >= len_thr])
    pred_list = tech_list & voc
    recall = len(pred_list) / len(tech_list)
    pred = len(pred_list) / len(word_list) if len(word_list) > 0 else 0
    f1 = 2 * recall * pred / (recall + pred) if recall + pred > 0 else 0
    print(len(pred_list), len(tech_list))
    return {'word_rec': "%.2f%%" % (recall * 100),
            'word_pred': "%.2f%%" % (pred * 100),
            'word_f1': "%.2f%%" % (f1 * 100),
            'voc_size': str(len(voc))}


# 指定长度
def get_dict_result_len(word_list, tech_dict, word_dict, freq_thr, len_thr):
    tech_list = set([word for word, attr in tech_dict.items()
                     if word_dict[word] >= freq_thr and len(word) == len_thr])
    word_list = set(word_list)
    voc = set([word for word in word_list
                if word_dict.get(word, 0) >= freq_thr and len(word) == len_thr])
    pred_list = tech_list & voc
    recall = len(pred_list) / len(tech_list)
    pred = len(pred_list) / len(voc) if len(voc) > 0 else 0
    f1 = 2 * recall * pred / (recall + pred) if recall + pred > 0 else 0
    print(len(pred_list), len(tech_list))
    return {'word_rec': "%.2f%%" % (recall * 100),
            'word_pred': "%.2f%%" % (pred * 100),
            'word_f1': "%.2f%%" % (f1 * 100),
            'voc_size': str(len(voc))}


def get_result_with_type(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
                         sample_sent_type_list, word_list, word_dict, freq_thr, len_thr,
                         type_list, eval_list, mode=1):
    score_dict = {}
    score_line = []
    print('total')
    score_dict1 = get_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode)
    tech_dict = get_tech_dict(sample_sent_seg_result_list, word_dict)
    score_dict2 = get_dict_result(word_list, tech_dict, word_dict, freq_thr, len_thr)
    score_dict1 = dict(score_dict1, **score_dict2)  # 合并
    score_dict['total'] = [score_dict1[eval] for eval in eval_list]
    score_line.extend([score_dict1[eval] for eval in eval_list])
    for type in type_list:
        print(type)
        sample_sent_idx_list_tmp = [idx for idx, type_tmp in zip(sample_sent_idx_list, sample_sent_type_list) if type_tmp == type]
        sample_sent_seg_result_list_tmp = [seg for seg, type_tmp in zip(sample_sent_seg_result_list, sample_sent_type_list) if type_tmp == type]
        score_dict_tmp = get_result(seg_text_dir, sample_sent_idx_list_tmp, sample_sent_seg_result_list_tmp, mode)
        # tech_dict_tmp = get_tech_dict(sample_sent_seg_result_list_tmp, word_dict)
        # score_dict_tmp2 = get_dict_result(word_list, tech_dict_tmp, word_dict, freq_thr, len_thr)
        # score_dict_tmp = dict(score_dict_tmp, **score_dict_tmp2)  # 合并
        score_tmp = [score_dict_tmp[eval] for eval in eval_list if eval in score_dict_tmp]
        score_dict[type] = score_tmp
        score_line.extend(score_tmp)
    return score_line, score_dict


def get_result_matrix(matrix, name,
                      sample_sent_idx_list, sample_sent_seg_result_list,
                      kappa_d_list, kappa_s_list, eval_list, seg_dir, mode=1):
    for kappa_d in kappa_d_list:
        for kappa_s in kappa_s_list:
            seg_text_dir = join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}', 'segmented_text.txt')
            score_dict_tmp = get_result(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list, mode)
            score_list = [score_dict_tmp[eval] for eval in eval_list]
            matrix.append([name, mode, kappa_d, kappa_s] + score_list)


from utils_seg import  get_dictionary_from_word_dict

def get_result_matrix_with_type(matrix, name,
                                sample_sent_idx_list, sample_sent_seg_result_list, sample_sent_type_list,
                                word_dict, freq_thr, len_thr,
                                kappa_d_list, kappa_s_list, type_list, eval_list,
                                seg_dir, mode=1):
    for kappa_d in kappa_d_list:
        for kappa_s in kappa_s_list:
            word_list = get_dictionary_from_tw_seg_result(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}'),
                                                          sample_sent_idx_list, sample_sent_seg_result_list, mode)
            seg_text_dir = join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}', 'segmented_text.txt')
            score_list, score_dict = get_result_with_type(seg_text_dir, sample_sent_idx_list, sample_sent_seg_result_list,
                                                          sample_sent_type_list, word_list, word_dict, freq_thr, len_thr,
                                                          type_list, eval_list, mode)
            matrix.append([name, mode, kappa_d, kappa_s] + score_list)


###########################################################
# evaluate 2

def get_sent_df_result(seg_data, poem_data,
                       sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list,
                       mode=1):  # mode 1 single; mode 2 double

    # check:
    seg_data = [seg.strip() for seg in seg_data]
    if mode == 2:
        seg_data = cut_double_line(seg_data)
    pred_seg_data = [seg_data[ix] for ix in sample_sent_idx_list]
    check_seg_poem(pred_seg_data, sample_sent_seg_result_list)

    poem_data_dict = {data['id']: data for data in poem_data}

    matrix = []
    for ix, seg_pred in enumerate(pred_seg_data):
        # id，原文，言，句， 标注文本，模板类型，预测文本，正确间隔数量，总间隔数量，是否完全正确
        sent_id = sample_sent_idx_list[ix]
        poem_id = sample_poem_idx_list[ix]
        data = poem_data_dict[poem_id]
        yan = data['yan']
        ju = data['ju']
        seg_true = sample_sent_seg_result_list[ix]
        text = seg_true.replace('/', '')
        template = ''.join([str(len(t)) for t in seg_true.split('/')])
        pred_01 = seg_to_01(seg_pred)[:-1]
        true_01 = seg_to_01(seg_true, '/')[:-1]
        pred_temp = ''.join([str(len(t)) for t in seg_pred.split(' ')])
        count = [sum([a==b for a, b in zip(pred_01, true_01)]), len(true_01)]
        tmp = [sent_id, text, yan, ju, seg_true, template, seg_pred, pred_temp, count[0], count[1], count[0]/count[1], count[0]==count[1]]
        matrix.append(tmp)

    df = pd.DataFrame(matrix, columns=['sent_id', 'text', 'yan', 'ju', 'true_seg', 'template', 'pred_seg', 'pred_temp',
                                       'pred_true', 'length', 'rate', 'all'])
    return df


def get_whole_df_result(poem_data,
                        sample_sent_idx_list, sample_sent_seg_result_list, sample_poem_idx_list,
                        kappa_d_list=[1e-5], kappa_s_list=[0.1, 0.01, 0.001, 0.0001, 1e-5],
                        seg_dir=join(output_dir, '20221013', 'mix23_single'), mode=1):

    for kappa_d in kappa_d_list:
        for kappa_s in kappa_s_list:
            seg_data = read_txt_encoding(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}', 'segmented_text.txt'))
            seg_data = [seg.strip() for seg in seg_data]
            df_tmp = get_sent_df_result(seg_data, poem_data, sample_sent_idx_list, sample_sent_seg_result_list,
                                        sample_poem_idx_list, mode)
            df_tmp.to_excel(join(seg_dir, f'topwords_seg_{kappa_d}_{kappa_s}', 'sent_result.xlsx'))


if __name__ == '__main__':

    pass


