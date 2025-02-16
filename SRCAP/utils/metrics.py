from nltk.translate.bleu_score import sentence_bleu,corpus_bleu, SmoothingFunction
import numpy as np
from multiprocessing import Pool

import nltk
import os
import random
from nltk.translate.bleu_score import SmoothingFunction

#from metrics.basic import Metrics
from abc import abstractmethod

def calculate_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction().method4

    # Flattening the nested lists (if they are lists of lists)
    reference = [token for sublist in reference for token in sublist] if isinstance(reference[0], list) else reference
    hypothesis = [token for sublist in hypothesis for token in sublist] if isinstance(hypothesis[0], list) else hypothesis
    
    # Calculate BLEU score
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

# def calculate_bleu_score(reference, hypothesis):
#     smoothie = SmoothingFunction()  # Smoothing method for BLEU score calculation
    
#     # Convert reference and hypothesis sequences to lists of strings
#     reference = [[str(token) for token in ref] for ref in reference]
#     hypothesis = [[str(token) for token in hyp] for hyp in hypothesis]

#     # Calculate BLEU score
#     return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)#,weights = (0.5, 0.5, 0.0, 0.

def psnr(psnr_real_targets, psnr_generated_targets):
    mse_max = 1
    mse = (psnr_real_targets - psnr_generated_targets) ** 2
    mse = mse.detach().cpu().numpy()
    mse_value = np.sqrt(mse.mean())
    psnr_picture = 20 * np.log10(mse_max / mse_value)
    return psnr_picture 



class Metrics:
    def __init__(self, name='Metric'):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class BLEU(Metrics):
    def __init__(self, name=None, test_text=None, real_text=None, gram=3, portion=1, if_use=False):
        assert type(gram) == int or type(gram) == list, 'Gram format error!'
        super(BLEU, self).__init__('%s-%s' % (name, gram))

        self.if_use = if_use
        self.test_text = test_text
        self.real_text = real_text
        self.gram = [gram] if type(gram) == int else gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_score(self, is_fast=True, given_gram=None):
        """
        Get BLEU scores.
        :param is_fast: Fast mode
        :param given_gram: Calculate specific n-gram BLEU score
        """
        if not self.if_use:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast(given_gram)
        return self.get_bleu(given_gram)

    def reset(self, test_text=None, real_text=None):
        self.test_text = test_text if test_text else self.test_text
        self.real_text = real_text if real_text else self.real_text

    def get_reference(self):
        reference = self.real_text.copy()

        # randomly choose a portion of test data
        # In-place shuffle
        random.shuffle(reference)
        len_ref = len(reference)
        reference = reference[:int(self.portion * len_ref)]
        self.reference = reference
        return reference

    def get_bleu(self, given_gram=None):
        if given_gram is not None:  # for single gram
            bleu = list()
            reference = self.get_reference()
            weight = tuple((1. / given_gram for _ in range(given_gram)))
            for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                bleu.append(self.cal_bleu(reference, hypothesis, weight))
            return round(sum(bleu) / len(bleu), 3)
        else:  # for multiple gram
            all_bleu = []
            for ngram in self.gram:
                bleu = list()
                reference = self.get_reference()
                weight = tuple((1. / ngram for _ in range(ngram)))
                for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                    bleu.append(self.cal_bleu(reference, hypothesis, weight))
                all_bleu.append(round(sum(bleu) / len(bleu), 3))
            return all_bleu

    @staticmethod
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self, given_gram=None):
        reference = self.get_reference()
        if given_gram is not None:  # for single gram
            return self.get_bleu_parallel(ngram=given_gram, reference=reference)
        else:  # for multiple gram
            all_bleu = []
            for ngram in self.gram:
                all_bleu.append(self.get_bleu_parallel(ngram=ngram, reference=reference))
            return all_bleu

    def get_bleu_parallel(self, ngram, reference):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
            result.append(pool.apply_async(self.cal_bleu, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return round(score / cnt, 3)