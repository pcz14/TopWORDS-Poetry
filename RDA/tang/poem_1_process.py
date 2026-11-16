# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 15:30
# @Author  : Changzai Pan
# @File    : wiki_1_gettext.py

from utils_seg import is_chinese_character, output_json, read_txt_encoding, get_sent_string_list_from_text_string, load_json
from RDA.tang.poem_0_config import *
import pandas as pd
from collections import Counter
# 数据预处理

def get_poem_data(text_string):
    punctuations = [char for char in set(''.join(text_string)) if not is_chinese_character(char)]
    poem_data = []
    sent_record = []
    for ix, text in enumerate(text_string):
        data_tmp = {}
        data_tmp['id'] = ix
        data_tmp['text'] = text
        sent_list = [sent for sent in get_sent_string_list_from_text_string(text, punctuations)
                                 if sent not in punctuations]
        data_tmp['sent_list'] = sent_list
        if '|'.join(sent_list) not in sent_record:
            data_tmp['is_repeat'] = False
            sent_record.append('|'.join(sent_list))
        else:
            data_tmp['is_repeat'] = True

        len_list = [len(sent) for sent in sent_list]
        data_tmp['regular'] = all(l == len_list[0] for l in len_list)
        data_tmp['yan'] = len_list[0] if data_tmp['regular'] else -1
        data_tmp['ju'] = len(data_tmp['sent_list'])
        poem_data.append(data_tmp)
    return poem_data


def get_word_candidate(text_string):
    word_candidate = []
    for sent in text_string:
        sent_tmp = sent.strip()
        if len(sent_tmp) == 5:
            word_candidate.extend([sent_tmp[:2], sent_tmp[2], sent_tmp[3:5], sent_tmp[2:4], sent_tmp[4]])
        if len(sent_tmp) == 7:
            word_candidate.extend([sent_tmp[:2], sent_tmp[2:4], sent_tmp[4], sent_tmp[5:7], sent_tmp[4:6], sent_tmp[6]])
    word_candidate_set = set(word_candidate)
    return word_candidate_set

# 只找前2-3的2和2-2-3的2-2
def get_word_candidate2(text_string):
    word_candidate = []
    for sent in text_string:
        sent_tmp = sent.strip()
        if len(sent_tmp) == 5:
            word_candidate.extend([sent_tmp[:2]])
        if len(sent_tmp) == 7:
            word_candidate.extend([sent_tmp[:2], sent_tmp[2:4]])
    word_candidate_set = set(word_candidate)
    return word_candidate_set


def output_modify_text_57(data_dir, poem_data_57):
    with open(join(data_dir, 'modify_text57.txt'), 'w', encoding='utf8') as f:
        for data in poem_data_57:
            for sent in data['sent_list']:
                f.write(sent + '\n')

    with open(join(data_dir, 'modify_text572.txt'), 'w', encoding='utf8') as f:
        for data in poem_data_57:
            for ix, sent in enumerate(data['sent_list']):
                if ix % 2 == 0:
                    f.write(sent)
                else:
                    f.write(sent + '\n')


def get_word_from_cbdb(data_dir, text_string):
    whole_string = ''.join(text_string)
    df_name = pd.read_csv(
        join(data_dir, 'named-entities-for-premodern-chinese-history-research-main', 'cbdb_entity_names.csv'))
    name_list = [name for name in df_name['name'] if len(name) <= 5 and name in whole_string]

    df_entr = pd.read_csv(
        join(data_dir, 'named-entities-for-premodern-chinese-history-research-main', 'cbdb_entity_entries.csv'))
    entr_list = [name for name in df_entr['c_entry_desc_chn'] if len(name) <= 5 and name in whole_string]

    df_addr = pd.read_csv(
        join(data_dir, 'named-entities-for-premodern-chinese-history-research-main', 'cbdb_entity_addresses.csv'))
    addr_list = [name for name in df_addr['addr_chn'] if len(name) <= 5 and name in whole_string]

    df_offi = pd.read_csv(
        join(data_dir, 'named-entities-for-premodern-chinese-history-research-main', 'cbdb_entity_offices.csv'))
    offi_list = [name for name in df_offi['office_name'] if len(name) <= 5 and name in whole_string]

    print('name', len(name_list))
    print('entry', len(entr_list))
    print('address', len(addr_list))
    print('office', len(offi_list))

    return list(set(name_list + entr_list + addr_list + offi_list))


def get_word_from_diangu(data_dir, text_string):
    whole_string = ''.join(text_string)
    data = read_txt_encoding(join(data_dir, '全唐诗典故辞典.txt'))
    return [word.strip() for word in data if word in whole_string]



def main():
    # 处理诗歌文本
    text_string = read_txt_encoding(join(data_dir, 'text_raw.txt'))  # 读取txt文本
    df_poem = pd.DataFrame({'id': range(len(text_string)), '诗歌': text_string})
    df_poem.to_excel(join(data_dir, 'poem_raw.xlsx'), index=False)  # 输出excel

    # 诗歌转化为data，以诗歌为单位，记录了诗歌的id，文本，是否与前诗重复，几言，几句，是否整齐(regular)
    poem_data = get_poem_data(text_string)
    output_json(poem_data, join(data_dir, 'poem_data.json'))

    # 只保留 五言七言 4/8句 诗歌，每行 为 两句诗
    poem_data_57 = [data for data in poem_data
                    if data['yan'] in [5, 7] and data['ju'] in [4, 8] and not data['is_repeat']]
    output_json(poem_data_57, join(data_dir, 'poem_data_57.json'))
    output_modify_text_57(data_dir, poem_data_57)

    # 按照212/221模板提取可能词汇
    word_candidate_set = get_word_candidate(read_txt_encoding(join(data_dir, 'modify_text57.txt')))
    with open(join(data_dir, 'word_candidate.txt'), 'w', encoding='utf8') as f:
        for word in word_candidate_set:
            _ = f.write(word + '\n')

    # 按照212/221模板提取可能词汇, 但是只提取前2的词
    word_candidate_set = get_word_candidate2(read_txt_encoding(join(data_dir, 'modify_text57.txt')))
    with open(join(data_dir, 'word_candidate2.txt'), 'w', encoding='utf8') as f:
        for word in word_candidate_set:
            _ = f.write(word + '\n')

    # 从cbdb提取词汇
    word_cbdb = get_word_from_cbdb(data_dir, text_string)
    word_diangu = get_word_from_diangu(data_dir, text_string)
    word_prior = list(set(word_cbdb + word_diangu))
    print(len(word_prior))
    with open(join(data_dir, 'prior_words_tang.txt'), 'w', encoding='utf8') as f:
        for word in word_prior:
            f.write(word + '\n')

    word_candidate = read_txt_encoding(join(data_dir, 'word_candidate.txt'))
    word_candidate = [w.strip() for w in word_candidate]
    with open(join(data_dir, 'prior_words_full.txt'), 'w', encoding='utf8') as f:
        for word in word_candidate:
            f.write(word + '\n')
        for word in word_prior:
            f.write(word + ' 10' + '\n')

    #
    word_prior = read_txt_encoding(join(data_dir, 'prior_words_tang.txt'))
    with open(join(data_dir, 'prior_words_tang.txt'), 'w', encoding='utf8') as f:
        for word in word_prior:
            f.write(word.strip() + ' ' + '10' + '\n')

    # 把候选词和cbdb中的词合并，并且保护起来
    word_candidate = read_txt_encoding(join(data_dir, 'word_candidate.txt'))
    word_candidate = [w.strip() for w in word_candidate]
    word_cbdb = read_txt_encoding(join(data_dir, 'prior_words_cbdb.txt'))
    word_cbdb = [w.strip() for w in word_cbdb]
    with open(join(data_dir, 'word_cand_cbdb.txt'), 'w', encoding='utf8') as f:
        for word in word_candidate + word_cbdb:
            f.write(word.strip() + '\n')

    # 统计数据中的词汇总量，5言算3个词，7言算4个词
    poem_data_57 = read_txt_encoding(join(data_dir, 'modify_text57.txt'))
    word_num = 0  # 765368
    for line in poem_data_57:
        if len(line.strip()) == 5:
            word_num += 3
        elif len(line.strip()) == 7:
            word_num += 4
        else:
            print(line)


if __name__ == '__main__':

    main()

