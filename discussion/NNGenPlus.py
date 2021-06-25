# encoding=utf-8

import os
import re
import time
import math
import random
import pickle
import argparse
import numpy as np
import logging.config
from tqdm import tqdm
from typing import List
import multiprocessing as mp
from functools import partial
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S'}},
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'}},
    'loggers': {'': {'handlers': ['default']}}
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


def read_to_list(filename):
    f = open(filename, 'r')
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        res.append(row.split())
    return res

def load_data(multi_path, n = -1, seed = 1):
    lines = list()
    if n == -1:
        for path in multi_path:
            lines += load_data_single(path, -1, seed)
    else:
        for path in multi_path:
            lines += load_data_single(path, int(n/len(multi_path)), seed)
    return lines

def load_data_single(path, n = -1, seed = 1):
    random.seed(seed)
    if n == -1:
        logger.info("\tload lines from ...{}".format(path[-50:]))
    else:
        logger.info("\tload {} lines from ...{}".format(n, path[-50:]))
    if re.match("\S+(.pkl|.pickle)$", path):
        lines = pickle.load(open(path, 'rb'))
        lines = [l.strip() for l in lines]
        if n != -1:
            lines = random.sample(lines,n)
        return lines
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
        lines = [l.strip() for l in lines]
        if n != -1:
            lines = random.sample(lines,n)
    return lines

def find_mixed_nn_single(test_diff_idx_test_simi, train_diffs, test_diffs, bleu_thre, smooth_function):
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    test_diff_idx, test_simi = test_diff_idx_test_simi
    candidates = test_simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    smooth_f = [SmoothingFunction().method0, SmoothingFunction().method1, SmoothingFunction().method2, SmoothingFunction().method3, SmoothingFunction().method4, SmoothingFunction().method5, SmoothingFunction().method6, SmoothingFunction().method7]
    for j in candidates:
        score = sentence_bleu([train_diffs[j].split()], test_diffs[test_diff_idx].split(),smoothing_function=smooth_f[smooth_function])
        if score > max_score:
            max_score = score
            max_idx = j
    return max_idx

def nngen(train_diffs, train_msgs, test_diffs, bleu_thre=5, max_dict_features=100000, gram_n = 1, smooth_function = 0):
    """NNGen
    NOTE: currently, we haven't everage GPU through
    pytorch or other libraries to optmize this function.
    """
    counter = CountVectorizer(max_features = max_dict_features, ngram_range=(1, gram_n))
    train_diffs_len = len(train_diffs)
    logger.info("Get train_matrix...{}".format(train_diffs_len))
    train_matrix = counter.fit_transform(train_diffs)
    logger.info("Get test_matrix...{}".format(len(test_diffs)))
    test_matrix = counter.transform(test_diffs)
    logger.info("train_matrix.get_shape(): {}, test_matrix.get_shape(): {}".format(train_matrix.get_shape(),test_matrix.get_shape()))
    
    logger.info("Get cosine similarity...")
    ## split the training data
    cores_n = int(mp.cpu_count())
    chunk_len = int(train_diffs_len / cores_n - 1)
    chunk_n = math.ceil(train_diffs_len / chunk_len)
    for i in range(0, train_diffs_len, chunk_len):
        if i + chunk_len >= train_diffs_len:
            similarities = np.concatenate((similarities, cosine_similarity(test_matrix, train_matrix[np.array(range(i,train_diffs_len))])), axis=1)
            cosine_similarity(test_matrix, train_matrix[np.array(range(i,train_diffs_len))])
        else:
            if i == 0:
                similarities = cosine_similarity(test_matrix, train_matrix[np.array(range(i,i+chunk_len))])
            else:
                similarities = np.concatenate((similarities, cosine_similarity(test_matrix, train_matrix[np.array(range(i,i+chunk_len))])), axis=1)
    del train_matrix, test_matrix
    
    logger.info("Retrieval...")
    pool = mp.Pool(cores_n)
    partial_func = partial(find_mixed_nn_single, train_diffs = train_diffs, test_diffs = test_diffs, bleu_thre=bleu_thre, smooth_function = smooth_function)
    max_idx_list = pool.map(partial_func, enumerate(similarities))
    pool.close()
    pool.join()
    test_msgs = [train_msgs[max_idx] for max_idx in max_idx_list]
    return test_msgs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NNGen')

    parser.add_argument("--train_n", type=int, default = -1, required = False)
    
    parser.add_argument("--test_n",type=int, default = -1, required = False)
    
    parser.add_argument("--seed_n", type=int, default = 0, required = False)
    
    parser.add_argument("--max_dict_features", type=int, default = 1000000, required = False)
    
    parser.add_argument("--gram_n", type=int, default = 1, required = False)
    
    parser.add_argument("--bleu_thre", type=int, default = 5, required = False)
    
    parser.add_argument("--smooth_function", type=int, default = 0, choices = range(8), required = False)
    
    parser.add_argument("--out_file", default = None, required = False)
    
    parser.add_argument("--test_msg_file", default = None, required = False)
    
    parser.add_argument("--test_diff_file", required = True)
    
    parser.add_argument("--train_diff_file", nargs='+', default = "", required = True)

    parser.add_argument("--train_msg_file", nargs='+', default = "", required = True)

    args = parser.parse_args()
    
    if args.out_file == None:
        args.out_file = "log/gram_{}/smooth_method_{}/gen.msg".format(args.gram_n, args.smooth_function)
        
    logger.info(args)
    
    if os.path.exists(args.out_file):
        logger.info("{} already exists.".format(args.out_file))
    else:
        start_time = time.time()
        logger.info("Load train diff data...")
        train_diffs = load_data(args.train_diff_file, args.train_n, args.seed_n)
        logger.info("Load train msg data...")
        train_msgs = load_data(args.train_msg_file, args.train_n, args.seed_n)
        logger.info("Load test diff data...")
        test_diffs = load_data_single(args.test_diff_file, args.test_n)
        logger.info("Start NNGen...")
        out_msgs = nngen(train_diffs, train_msgs, test_diffs, bleu_thre=args.bleu_thre, max_dict_features=args.max_dict_features, gram_n = args.gram_n, smooth_function = args.smooth_function)
        logger.info("End NNGen...")
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, 'w') as out_f:
            out_f.write("\n".join(out_msgs) + "\n")
        time_cost = time.time() - start_time
        logger.info("File saved in {}, cost {}s".format(args.out_file, time_cost))