
from typing import List, Set, Dict
import numpy as np

from TopwordsSegMix2.utils import Word, get_uint_dtype


class Dictionary:
    def initialize_word_dictionary(self,
                                   word_list: List[Word],
                                   word_raw_count: List[int],
                                   active_sent_num: int,
                                   prior_word_freq: Dict[Word, int],
                                   protect_word_set: Set[Word] = set()):
        # check
        protect_word_set = set(word_list) & protect_word_set
        assert len(word_list) == len(word_raw_count), 'Input word_num arguments are not consistent.'
        assert protect_word_set.issubset(word_list), 'protect_word_set is not a subset of word_list'

        # word
        self.word_list = word_list + ['']
        self.word_num = len(self.word_list)
        self.word_len = np.fromiter((len(word) for word in self.word_list), dtype=np.uint64, count=self.word_num)
        # np.fromiter: word_list的前count个
        self.word_max_len = int(self.word_len.max())
        self.word_len = self.word_len.astype(get_uint_dtype(self.word_max_len))
        self.word2ix = {word: ix for ix, word in enumerate(self.word_list)}
        
        self.word_prior_count = {self.word2ix[word]: freq for word, freq in prior_word_freq.items() if freq > 0 and word in self.word_list}
        self.word_raw_count = np.array(word_raw_count + [0], dtype=np.uint64)
        self.word_raw_count = self.word_raw_count.astype(get_uint_dtype(self.word_raw_count.max()))

        self.word_bool = np.ones(self.word_num, dtype=np.bool_)
        self.word_protect_bool = (self.word_len <= 1) | np.array([word in protect_word_set for word in self.word_list])

        self.theta_value = self.word_raw_count.astype(np.float64)
        self.theta_value[-1] = active_sent_num
        self.theta_value /= self.theta_value.sum()

    def initialize_component(self, comp_value=None, comp_k=1):
        if comp_value is not None:
            self.comp_value = comp_value
        else:
            self.comp_value = {l: np.ones(comp_k)/comp_k for l in range(1000)}

    def __init__(self):
        # word
        self.word_list = None
        self.word_num = None
        self.word_len = None
        self.word_max_len = None
        self.word2ix = None

        self.word_raw_count = None
        self.word_post_count = None
        self.word_post_mean_seg_count = None
        self.word_post_mode_seg_count = None

        self.word_bool = None
        self.word_protect_bool = None
        self.word_score = None

        self.theta_value = None
        self.theta_new = None

        # log likelihood
        self.log_likelihood = 0.0

        self.comp_value = {}  # {sent_length: [pi_0, .. pi_K]}, e.g. {5:[0.3, 0.7], 7:[0.5,0.5]}
        self.comp_new = {}
        self.comp_post_count = None

    def prune_by_count(self, prune_by_count_threshold: float):
        active_word_num = self.word_bool.sum()
        self.word_bool = self.word_protect_bool | (self.word_bool & (self.theta_new > prune_by_count_threshold))
        is_prune = True if self.word_bool.sum() < active_word_num else False
        return is_prune

    def prune_by_para(self, prune_by_para_threshold: float):
        active_word_num = self.word_bool.sum()
        self.word_bool = self.word_protect_bool | (self.word_bool & (self.theta_value > prune_by_para_threshold))
        for ix, theta in enumerate(self.theta_value):
            if self.word_protect_bool[ix] and theta < prune_by_para_threshold:
                self.theta_value[ix] = prune_by_para_threshold # 保护的value最低是prune_by_para_threshold
        is_prune = True if self.word_bool.sum() < active_word_num else False
        return is_prune
        # for word_ix, word in enumerate(self.word_list):
        #     if self.word_bool[word_ix] and len(word) > 1:
        #         if self.theta_value[word_ix] < prune_by_para_threshold:
        #             self.word_bool[word_ix] = False

    def prune_by_score(self, prune_by_score_threshold: float):
        active_word_num = self.word_bool.sum()
        self.word_bool = self.word_protect_bool | (self.word_bool & (self.word_score > prune_by_score_threshold))
        is_prune = True if self.word_bool.sum() < active_word_num else False
        return is_prune
        # for word_ix, word in enumerate(self.word_list):
        #     if self.word_bool[word_ix] and len(word) > 1:
        #         if self.word_score[word_ix] < prune_by_score_threshold:
        #             self.word_bool[word_ix] = False
