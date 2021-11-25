from data_process_tools import pygment_mul_line, split_variable, line_filter_cmt
import multiprocessing as mp
from tqdm import tqdm
import logging.config
import numpy as np
import argparse
import pickle
import random
import json
import time
import os
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S'}},
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'}},
    'loggers': {'': {'handlers': ['default']}}
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# filter Chinese sentence
def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def pickle_load_single(path):
    logger.info("load from ...{}".format(path[-50:]))
    return pickle.load(open(path,"rb"))

class Data4CopynetV3:
    def __init__(self):
        # initial all variable
        # save msg data as list
        self.msgtext = []
        # save splited msg data as list
        self.msg = []
        # save diff generated code change as list
        self.difftext = []
        # split diff and save as list
        self.difftoken = []
        # split diff variable as diff attribution
        self.diffatt = []
        # + or - mark before a token or a line,
        # when it's length is smaller than diff token, it's marking a line.
        # when they're equal, it's marking a token
        self.diffmark = []
        # dict for both diff token and msg word
        self.word2index = {}
        # if a word can't be generated, set it's genmask to 0
        self.genmask = []
        # if a word can't be copy, set it's copymask to 0
        self.copymask = []
        # save the dict for entity and it's representation
        self.variable = []
        # save the word index of the first word that only appear in msg
        self.difftoken_start = 0
        # save the commit time
        self.time = []


    def build(self, filenames, single_language = True, one_block = False, language = ["java"], path = None, txt_file = False, need_time = False, use_cache_in_outdir = False):
        # this function read data in pickle file and fill msg*, diff*, variable
        # filenames is a list of many pickle files'name
        # single_language means only one certain language code change. boolean
        # one_block means only one diff block. boolean
        # data = json.load(open(filename, 'r'), encoding='utf-8') ## if it is a json file
        for lan in language:
            lan_cache_path = os.path.join(path, "cache", lan)
            if use_cache_in_outdir and os.path.exists(lan_cache_path) and not txt_file and (os.path.exists(os.path.join(lan_cache_path,"time.json")) or not need_time):
#                 logger.info("loading cache from {}".format(lan_cache_path))
                logger.info("Not loading cache from {}".format(lan_cache_path))
#                 self.add_data(lan_cache_path, time=need_time)
                logger.info("There are {} msgs, {} diffs, {} times".format(len(self.msg), len(self.difftext), len(self.time)))
            else:
                if txt_file:
                    diff_file_lst, msg_file_lst = filenames
                    data = list()
                    for i in range(len(diff_file_lst)):
                        diff_per_file = open(diff_file_lst[i],"r").read().split("\n")
                        msg_per_file = open(msg_file_lst[i],"r").read().split("\n")
                        if len(diff_per_file) == len(msg_per_file):
                            logger.info("there are {} commits in the {}st file".format(len(diff_per_file), i+1))
                            for commit_i in range(len(msg_per_file)):
                                commit = dict()
                                commit['diff'] = diff_per_file[commit_i]
                                commit['msg'] = msg_per_file[commit_i]
                                data.append(commit)
                        else:
                            logger.warning("{} diff file cannot match {} msg file".format(i, i))
                            exit()
                else:
                    data = list()
                    if filenames[0][-10:] == "/cache.pkl":
                        logger.info("load from {} cache...".format(lan))
                    else:
                        logger.info("load from repo raw {} data...".format(lan))
                        
                    this_lan_files = list()
                    for filename in filenames:
                        if os.path.dirname(filename).split("/")[-1] == lan:
                            this_lan_files.append(filename)
                        
                    this_lan_files = filenames # delete this line after 20210430
                    
                    logger.debug(this_lan_files)
                    
                    pool = mp.Pool(int(mp.cpu_count()))
                    repo_data_list = pool.map(pickle_load_single, this_lan_files)
                    pool.close()
                    pool.join()
                    for i in repo_data_list:
                        data += i
                    # if filenames[0][:-10] != "/cache.pkl":
                    #     pickle.dump(data, open(os.path.join(os.path.dirname(os.path.dirname(filenames)), "cache.pkl"),"wb"), protocol = 5)
                    del repo_data_list
                logger.info("load {} commits finished".format(len(data)))

                pattern = re.compile(r'\w+')    # for splitting msg
                count_none, count_ch = 0, 0
                cnt =0
                with tqdm(total=len(data), desc="build") as pbar:
                    for x, i in enumerate(data):
                        if x > 1000000000:  # x for debug, set value of x to a small num
                            break
                        diff = i['diff']
                        if diff == None or i['msg'] == None:
                            count_none+=1
                            pbar.update(1)
                            continue
                        if is_contain_chinese(diff) or is_contain_chinese(i['msg']):
                            count_ch+=1
                            pbar.update(1)
                            continue
                        if txt_file:
                            diff = diff.replace("<nl> ", "\n")
                            diff = diff.replace("ppp", "+++")
                            diff = diff.replace("mmm", "---")
                            files = diff.count('+++ ')
                        else:
                            files = diff.count('diff --git')
                        language_file = {"java":".java", "python":".py", "cpp":".cpp", "javascript":".js", "csharp":".cs"}
                        if txt_file:
                            language_file = {"java":" . java ", "python":" . py ", "cpp":" . cpp ", "javascript":" . js ", "csharp":" . cs "}
                        
                        if txt_file:
                            single_language_files = diff.count(language_file[lan]) // 2
                        else:
                            single_language_files = diff.count(language_file[lan]) // 4
                        
                        if txt_file:
                            blocks = 1
                        else:
                            blocks = diff.count('@@') // 2
                        if single_language and (files < 1 or files != single_language_files):
                            pbar.update(1)
                            continue
                        if one_block and blocks != 1:
                            pbar.update(1)
                            continue
                        ls = diff.splitlines()
                        single_language_lines = list()
                        diff_marks = list()
                        other_file = False
                        for line in ls:
                            if len(line) < 1: # blank line
                                continue
                            if line.startswith('+++') or line.startswith('---'):
                                if not line.endswith(language_file[lan]):
                                    other_file = True
                                    break
                                continue
                            st = line[0]
                            line = line[1:].strip()
                            if not txt_file and st == '@':
                                single_language_lines.append('NewBlock ' + line[line.find('@@') + 3:].strip())
                                diff_marks.append(2)
                            elif not txt_file and st == ' ': # the code not changed 
                                single_language_lines.append(line_filter_cmt(line, lan))
                                diff_marks.append(2)
                            elif txt_file and st != '+' and st != '-': # the code not changed 
                                single_language_lines.append(line_filter_cmt(line, lan))
                                diff_marks.append(2)
                            elif st == '-': # the code deleted
                                single_language_lines.append(line_filter_cmt(line, lan))
                                diff_marks.append(1)
                            elif st == '+': # the code added
                                single_language_lines.append(line_filter_cmt(line, lan))
                                diff_marks.append(3)
                            # if len(single_language_lines) > 0:
                            #     ttt = len(single_language_lines)
                            #     logger.info(single_language_lines[ttt-1])
                        if other_file:
                            pbar.update(1)
                            continue
                        # logger.info("single_language_lines:", single_language_lines)
                        try:
                            tokenList, varDict = pygment_mul_line(single_language_lines, lan)
                        except:
                            print(single_language_lines)
                            cnt = cnt+1
                        msg = pattern.findall(i['msg'])
                        msg = [i for i in msg if i != '' and not i.isspace()]

                        self.msgtext.append(i['msg'])
                        self.msg.append(msg)
                        self.difftext.append(diff)
                        # length of diff token and diff mark aren't equal
                        self.difftoken.append(tokenList)
                        self.diffmark.append(diff_marks)
                        self.variable.append(varDict)
                        # logger.info("No. message", i['msgs'],msg,)# x, files, single_language_files, blocks)
                        if need_time:
                            self.time.append(i['date'])
                        pbar.update(1)
                logger.info("%d error"%cnt)
                logger.info("{} commits in {} are None; {} commits are in Chinese".format(count_none, lan, count_ch))
                logger.info("saving cache in {}".format(lan_cache_path))
                os.makedirs(lan_cache_path, exist_ok = True)
                self.save_data(lan_cache_path, save_difftext=True, save_msgtext=True, save_diff=True, save_msg=True,
                               save_variable=True, save_word2index=False, save_time=need_time)


    def re_split_msg(self):
        # to split msg in case build function are wrong
        msgs = list()
        pattern = re.compile(r'\w+')
        for i in self.msgtext:
            msg = pattern.findall(i)
            msg = [j for j in msg if j != '' and not j.isspace()]
            msgs.append(msg)
        self.msg = msgs

    def re_process_diff(self):
        # split variable in diff and save it into diff attribution
        # this function should be invoked after build only once
        diff_tokens, diff_marks, diff_atts = [], [], []
        for i, j, k in zip(self.difftoken, self.diffmark, self.variable):
            diff_token, diff_att = [], []
            for x in i:
                if x in k:
                    diff_att.append(split_variable(x))
                    diff_token.append(x)
                else:
                    diff_att.append([])
                    diff_token.append(x)
            diff_mark, diff_token, diff_att = self.mark_token(j, diff_token, diff_att) ###
            # translate line mark into token mark
            diff_tokens.append(diff_token)
            diff_marks.append(diff_mark)
            diff_atts.append(diff_att)
        self.diffmark = diff_marks
        self.difftoken = diff_tokens
        self.diffatt = diff_atts

    def mark_token(self, marklist, tokenlist, attlist):
        lineNum = 0
        diff_mark = list()
        for i in tokenlist:
            if lineNum < len(marklist):
                diff_mark.append(marklist[lineNum]) ###
            else:
                diff_mark.append(2) ###
                # logger.debug(marklist, tokenlist, attlist)
            if i == '<nl>':
                lineNum += 1
        while lineNum < len(marklist):
            diff_mark.append(marklist[lineNum])
            tokenlist.append('<nl>')
            attlist.append([])
            lineNum += 1
        return diff_mark, tokenlist, attlist

    def filter(self, min_diff, max_diff, min_msg, max_msg):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        # delte `, self.diffatt`
        for i, j, k, l, m, n in zip(self.difftoken, self.diffmark, self.msg, self.variable, self.difftext, self.msgtext):
        # for i, j, k, l, m, n, o in zip(self.difftoken, self.diffmark, self.msg, self.variable, self.difftext,
        #                                        self.msgtext, self.diffatt):
            diff = []
            for idx, d in enumerate(i):
                if (d == '<nb>' or d == '<nl>') and idx + 1 < len(j) and j[idx + 1] != 2:
                    diff.append(d)
                    if j[idx + 1] == 1:
                        diff.append('-')
                    else:
                        diff.append('+')
                else:
                    diff.append(d)
            
            if min_diff <= len(diff) < max_diff and min_msg <= len(k) < max_msg:
                save = True
                for w in k:
                    # if (not w.isalpha() and w not in l) or w.lower() == 'flaw' or w.lower() == 'flaws':
                    if not w.isalpha() and w not in l:
                        save = False
                        break
                if save:
                    na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n)# , ng.append(o)
            # else:
            #     logger.info(len(diff),diff)
        self.difftoken, self.diffmark, self.msg, self.variable, self.difftext = na, nb, nc, nd, ne
        self.msgtext, self.diffatt = nf, ng


    def shuffle(self):
        random.seed(1)
        random.shuffle(self.difftoken)
        random.seed(1)
        random.shuffle(self.diffmark)
        random.seed(1)
        random.shuffle(self.msg)
        random.seed(1)
        random.shuffle(self.variable)
        random.seed(1)
        random.shuffle(self.difftext)
        random.seed(1)
        random.shuffle(self.msgtext)
        random.seed(1)
        random.shuffle(self.diffatt)


    def build_word2index1(self):
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # vp = re.compile(r'(n|a|f|c)\d+$')
        # cp = re.compile(r'(FLOAT|NUMBER|STRING)\d+$')
        v_set = set()
        word_count1, word_count2 = dict(), dict()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k in zip(self.difftoken, self.msg, self.variable):
            for x in j:
                if x in k:
                    continue
                x = x.lower()
                if x in lemmatization:
                    x = lemmatization[x]
                if x in word_count2:
                    word_count2[x] += 1
                else:
                    word_count2[x] = 1
            for x in i:
                if x in k:
                    v_set.add(k[x])
                    continue
                if x in word_count1:
                    word_count1[x] += 1
                else:
                    word_count1[x] = 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        num += 1
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1

    def gen_tensor1(self, start, end, vocab_size, diff_len=200, msg_len=20):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        mg = np.zeros([length, msg_len + 1])
        for i, (j, k, l, m) in enumerate(zip(diff, diff_m, msg, va)):
            for idx, (dt, dm) in enumerate(zip(j, k)):
                d_mark[i, idx] = dm
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, mg, genmask, copymask

    def build_word2index2(self, output_folder):
        # this function should be invoked after re_process_diff
        # but as later as possible
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # count word frequencies in diff and msg
        word_count1, word_count2, word_count3 = dict(), dict(), dict()
        v_set = set()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l in zip(self.difftoken, self.msg, self.diffatt, self.variable):
            for x in j:
                if x in l:
                    continue
                x = x.lower()
                x = lemmatization[x] if x in lemmatization else x
                word_count2[x] = word_count2[x] + 1 if x in word_count2 else 1
            for x in i:
                if x in l:
                    v_set.add(l[x])
                    continue
                word_count1[x] = word_count1[x] + 1 if x in word_count1 else 1
            for x in k:
                for y in x:
                    word_count3[y] = word_count3[y] + 1 if y in word_count3 else 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        word_count3 = sorted(word_count3.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        num += 1
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in word_count3:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        
        file_path = os.path.join(output_folder, 'kv.txt')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, 'w')
        for i in word_count1:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        for i in word_count2:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        for i in word_count3:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        # json.dump(self.word2index, open('data4mul_block/word2index.json', 'w'))

    def gen_tensor2(self, start, end, vocab_size, diff_len=200, attr_num=5, msg_len=20): # [update]
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        diff_a = self.diffatt[start: end]
        
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len, attr_num])
        mg = np.zeros([length, msg_len + 1])
        logger.debug(d_mark.shape, mg.shape) # (5000, 200)
        for i, (j, k, l, m, n) in enumerate(zip(diff, diff_m, msg, va, diff_a)):
            if len(j) > diff_len:
                logger.info("len_diff:{} {}".format(len(j),i))
            if len(k) > diff_len:
                logger.info("len_diff_m:{} {}".format(len(k),i))
            if len(l) > diff_len:
                logger.info("len_msg:{} {}".format(len(l),i))
            if len(m) > diff_len:
                logger.info("len_va:{} {}".format(len(m),i))
            if len(n) > diff_len:
                logger.info("len_diff_a:{} {}".format(len(n),i))
            # logger.debug(i, j, k, l, m, n)
            for idx, (dt, dm, da) in enumerate(zip(j, k, n)):
                # logger.debug(i,idx,dm)
                d_mark[i, idx] = dm
                # logger.debug(d_mark）
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
                for idx2, a in enumerate(da):
                    if idx2 >= attr_num:
                        break
                    d_attr[i, idx, idx2] = self.word2index[a] if a in self.word2index else self.word2index['<unkd>']
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, d_attr, mg, genmask, copymask

    def build_word2index3(self):
        # this function should be invoked after re_process_diff
        # but as later as possible
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # count word frequencies in diff and msg
        word_count1, word_count2, word_count3 = dict(), dict(), dict()
        v_set = set()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l in zip(self.difftoken, self.msg, self.diffatt, self.variable):
            for x in j:
                if x in l:
                    continue
                x = x.lower()
                x = lemmatization[x] if x in lemmatization else x
                word_count2[x] = word_count2[x] + 1 if x in word_count2 else 1
            for x in i:
                if x in l:
                    v_set.add(l[x])
                    continue
                word_count1[x] = word_count1[x] + 1 if x in word_count1 else 1
            for x in k:
                for y in x:
                    word_count3[y] = word_count3[y] + 1 if y in word_count3 else 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        word_count3 = sorted(word_count3.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        self.word2index['<sp>'] = num + 1
        self.word2index['<add>'] = num + 2
        self.word2index['<del>'] = num + 3
        num += 4
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in word_count3:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1

    def gen_tensor3(self, start, end, vocab_size, diff_len=200, msg_len=20):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        diff_a = self.diffatt[start: end]
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len])
        mg = np.zeros([length, msg_len + 1])
        for i, (j, k, l, m, n) in enumerate(zip(diff, diff_m, msg, va, diff_a)):
            num = 0
            for idx, (dt, dm, da) in enumerate(zip(j, k, n)):
                d_mark[i, idx] = dm
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
                if len(da) > 0:
                    if num + len(da) + 1 >= diff_len:
                        break
                    if dm == 2:
                        d_attr[i, num] = self.word2index['<sp>']
                    elif dm == 1:
                        d_attr[i, num] = self.word2index['<del>']
                    else:
                        d_attr[i, num] = self.word2index['<add>']
                    num += 1
                    for a in da:
                        d_attr[i, num] = self.word2index[a] if a in self.word2index else self.word2index['<unkd']
                        num += 1
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, d_attr, mg, genmask, copymask

    def save_data(self, path, save_difftext=False, save_msgtext=False, save_diff=False, save_msg=False,
                  save_variable=False, save_word2index=False, save_time=False):
        # you can do it any time, just save all data
        os.makedirs(os.path.join(path), exist_ok=True)
        if save_difftext:
            json.dump(self.difftext, open(os.path.join(path, 'difftext.json'), 'w'))
            # logger.debug("Completed " + "save_difftext")
        if save_msgtext:
            json.dump(self.msgtext, open(os.path.join(path, 'msgtext.json'), 'w'))
            # logger.debug("Completed " + "save_msgtext")
        if save_diff:
            json.dump(self.difftoken, open(os.path.join(path, 'difftoken.json'), 'w'))
            json.dump(self.diffmark, open(os.path.join(path, 'diffmark.json'), 'w'))
            json.dump(self.diffatt, open(os.path.join(path, 'diffatt.json'), 'w'))
            # logger.debug("Completed " + "save_diff")
        if save_msg:
            json.dump(self.msg, open(os.path.join(path, 'msg.json'), 'w'))
            # logger.debug("Completed " + "save_msg")
        if save_variable:
            json.dump(self.variable, open(os.path.join(path, 'variable.json'), 'w'))
            # logger.debug("Completed " + "save_variable")
        if save_word2index:
            json.dump(self.word2index, open(os.path.join(path, 'word2index.json'), 'w'))
            json.dump(self.genmask, open(os.path.join(path, 'genmask.json'), 'w'))
            json.dump(self.copymask, open(os.path.join(path, 'copymask.json'), 'w'))
            json.dump(self.difftoken_start, open(os.path.join(path, 'num.json'), 'w'))
            # logger.debug("Completed " + "save_word2index")
        if save_time:
            json.dump(self.time, open(os.path.join(path, 'time.json'), 'w'))

    def load_data(self, path, load_difftext=True, load_msgtext=True, load_diff=True, load_msg=True,
                  load_variable=True, load_word2index=True, load_time=False):
        # load data from disk
        if load_difftext:
            self.difftext = json.load(open(os.path.join(path, 'difftext.json')))
        if load_msgtext:
            self.msgtext = json.load(open(os.path.join(path, 'msgtext.json')))
        if load_diff:
            self.difftoken = json.load(open(os.path.join(path, 'difftoken.json')))
            self.diffmark = json.load(open(os.path.join(path, 'diffmark.json')))
            self.diffatt = json.load(open(os.path.join(path, 'diffatt.json')))
        if load_msg:
            self.msg = json.load(open(os.path.join(path, 'msg.json')))
        if load_variable:
            self.variable = json.load(open(os.path.join(path, 'variable.json')))
        if load_word2index:
            self.word2index = json.load(open(os.path.join(path, 'word2index.json')))
            self.genmask = json.load(open(os.path.join(path, 'genmask.json')))
            self.copymask = json.load(open(os.path.join(path, 'copymask.json')))
            self.difftoken_start = int(json.load(open(os.path.join(path, 'num.json'))))
        if load_time:
            self.time = json.load(open(os.path.join(path, 'time.json')))
    
    def add_data(self, path, difftext=True, msgtext=True, diff=True, msg=True,
                  variable=True, time=False):
        # load data from disk
        if difftext:
            logger.debug(len(self.difftext))
            self.difftext += json.load(open(os.path.join(path, 'difftext.json')))
            logger.debug(len(self.difftext))
        if msgtext:
            self.msgtext += json.load(open(os.path.join(path, 'msgtext.json')))
        if diff:
            self.difftoken += json.load(open(os.path.join(path, 'difftoken.json')))
            self.diffmark += json.load(open(os.path.join(path, 'diffmark.json')))
            self.diffatt += json.load(open(os.path.join(path, 'diffatt.json')))
        if msg:
            self.msg += json.load(open(os.path.join(path, 'msg.json')))
        if variable:
            self.variable += json.load(open(os.path.join(path, 'variable.json')))
        if time:
            self.time += json.load(open(os.path.join(path, 'time.json')))

    def data_constraint(self, min_diff, max_diff, min_msg, max_msg):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        for i, j, k, l, m, n, o in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                    self.msgtext, self.diffatt):
            if min_diff <= len(i) < max_diff and min_msg <= len(k) < max_msg:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(o)
        self.diffmark, self.difftoken, self.msg, self.variable, self. difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng

    def deduplication(self):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        diffset = set()
        for i, j, k, l, m, n, o in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                       self.msgtext, self.diffatt):
            iii = str(i)+''.join(j)+n if len(k) < 10 else n # [to-do] 10 need to update?
            if iii not in diffset:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(o)
                diffset.add(iii)
            # else:
            #     logger.debug("There is a duplication: ",iii)
        self.diffmark, self.difftoken, self.msg, self.variable, self.difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng

    def remove_unk(self): # [delete]
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l, m, n, p in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                    self.msgtext, self.diffatt):
            is_unk = False
            for o in k:
                if o.lower() in lemmatization:
                    continue
                if (o.lower() not in self.word2index and o not in l) or o.lower() == 'flaw' or o.lower() == 'flaws':
                    is_unk = True
                    break
            if not is_unk:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(p)
        self.diffmark, self.difftoken, self.msg, self.variable, self.difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng

    def gen_tensor(self, start, end, vocab_size, diff_len=300, msg_len=20, attribute_num=5):  # [delete]
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        difftoken = self.difftoken[start: end]
        diffmark = self.diffmark[start: end]
        diffatt = self.diffatt[start: end]
        variable = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len, attribute_num])
        mg = np.zeros([length, msg_len + 1])
        cp = np.zeros([length, msg_len, 2])
        vocab_mask = np.zeros([vocab_size, ])
        for i, (j, k, l, m, n) in enumerate(zip(diffmark, difftoken, msg, variable, diffatt)):
            for idx, (mk, tk, ak) in enumerate(zip(j, k, n)):
                d_mark[i, idx] = mk
                tkn = self.word2index[tk] if tk in self.word2index else 2
                d_word[i, idx] = tkn
                idx1 = 0
                for va in ak:
                    if idx1 >= attribute_num:
                        break
                    if va in self.word2index:
                        d_attr[i, idx, idx1] = self.word2index[va]
                        idx1 += 1
                while idx1 < attribute_num:
                    d_attr[i, idx, idx1] = 7
                    idx1 += 1
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                if c in m:
                    c0 = self.word2index[m[c]] if m[c] in self.word2index else 3
                    cp[i, idx, 1] = 1
                else:
                    c = c.lower()
                    if c in lemmatization:
                        c = lemmatization[c]
                    c0 = self.word2index[c] if c in self.word2index else 3
                    if c0 >= self.msg_word_start:
                        cp[i, idx, 0] = 1
                mg[i, idx + 1] = c0
        vocab_mask[0:len(self.genmask)] = self.genmask
        return d_mark, d_word, d_attr, mg, cp, vocab_mask

    def get_data(self, start, end, re_difftext=False, re_msgtext=False, re_diffmark=False, re_difftoken=False,
                 re_msg=False, re_variable=False):
        data = list()
        if re_difftext:
            data.append(self.difftext[start: end])
        if re_msgtext:
            data.append(self.msgtext[start: end])
        if re_diffmark:
            data.append(self.diffmark[start: end])
        if re_difftoken:
            data.append(self.difftoken[start: end])
        if re_msg:
            data.append(self.msg[start: end])
        if re_variable:
            data.append(self.variable[start: end])
        return data

    def get_word2index(self):
        return self.word2index, self.difftoken_start

    def get_info(self):
        # logger.debug(len(self.difftoken))
        sum1, sum2, num = 0, 0, 0
        for i, j in zip(self.difftoken, self.msg):
            sum1 += len(i)
            sum2 += len(j)
        logger.info("average length of difftoken is {}, msg is {}".format(sum1 / len(self.difftoken), sum2 / len(self.msg)))

if __name__ == '__main__':
    time_start=time.time()
    
    ##### get parameters #####
    parser = argparse.ArgumentParser(description='preprocess the raw_data to preprocessed_data')

    parser.add_argument("-rn","--repo_full_name", metavar="Theano/Theano", default = None,
                        help='the list of repositories\' fullnames of in GitHub. E.g. Theano/Theano ', required = False)

    parser.add_argument("-rl","--repo_list", metavar="top100/repo_list.json", default = None, # ".../repo_list.json",
                        help='the list of repositories\' fullnames of in GitHub. E.g. top100/repo_list.json ', required = False)

    parser.add_argument("-lan","--language", metavar="java", choices=["python", "javascript", "cpp", "java", "csharp"], nargs='+',
                        help='the language you want to deal with in the dataset', required = True)
    
    parser.add_argument("-in","--input_folder", metavar="raw_data/", default = None, # default=".../raw_data"
                        help='the folder which contains raw data(such as java/Activiti/Activiti.pickle)', required = False)
    
    parser.add_argument("-out","--output_folder", metavar="preprocessed_data/", default="preprocessed_data/",
                        help='the path of preprocessed_data data(such as *_Theano_Theano.json)', required = False)
    
    parser.add_argument("-s","--sort", metavar="random", default="random", choices=["random", "time","repo"],
                        help='the data is sorted by time or random', required = False)

    parser.add_argument("--min_diff", type=int, metavar=3, default=3,
                        help='the minimum number of tokens in each diff, int.', required = False)

    parser.add_argument("--max_diff", type=int, metavar=201, default=201,
                        help='the maximum number of tokens in each diff, int.', required = False)

    parser.add_argument("--min_msg", type=int, metavar=3, default=3,
                        help='the minimum number of tokens in each msg, int.', required = False)

    parser.add_argument("--max_msg", type=int, metavar=21, default=21,
                        help='the maximum number of tokens in each msg, int.', required = False)

    parser.add_argument("--cache", action='store_true', help='if you hava a `cache.pkl` in `input_folder/langugae`', required = False)

    
    parser.add_argument("--need_time", action='store_true', help='if you want to have `time.json` in `output_folder/langugae/...`', required = False)

    args = parser.parse_args()
    
    logger.info(args)
    ##### get parameters #####
       

    ##### have a cache #####
    if args.cache:
        logger.info("get {} caches ...".format(len(args.language)))
        repo_full_name = list()
        for lan in args.language:
            repo_full_name.append(os.path.join(args.input_folder, lan, "cache.pkl"))
    else: 
        ##### get repo pickle files' path #####
        repo_full_name = list()
        repo_list = json.load(open(args.repo_list,"r"))
        for lan in args.language:
            for root, dirs, files in os.walk(os.path.join(args.input_folder, lan)):
                for file in files:
                    if re.search(".pickle", file) != None:
                        path = os.path.join(root, file)
                        if path[len(args.input_folder) + len(lan) + 1 : -len(".pickle")] in repo_list[lan]:
                            repo_full_name.append(path)
        logger.info("get data from {} repos ...".format(len(repo_full_name)))
        ##### get repo pickle files' path #####
    logger.debug(repo_full_name)
    
    args.language.sort()
    multi_lan_str = "_".join(args.language)
    if args.repo_full_name != None:
        repo_str = "_".join(args.repo_full_name)
    else:
        repo_str = multi_lan_str
    output_folder = os.path.join(args.output_folder, "sort_{}-min_diff{}-max_diff{}-min_msg{}-max_diff{}".format(args.sort, args.min_diff, args.max_diff, args.min_msg, args.max_msg), multi_lan_str, repo_str)
    if os.path.exists(os.path.join(output_folder,"word2index.json")):
        logger.info("File already exists in {}".format(output_folder))
        os._exit(0)
    os.makedirs(output_folder, exist_ok=True)


    logger.info("Initialize ...")
    dataset = Data4CopynetV3()
    
    logger.info("Prepare to build ...")
    dataset.build(repo_full_name, language = args.language, path = args.output_folder, need_time = args.need_time)
    logger.debug("build {}".format(len(dataset.msgtext)))

    dataset.filter(args.min_diff, args.max_diff, args.min_msg, args.max_diff) # [to-do]
    logger.debug("filter {}".format(len(dataset.msgtext)))

    if args.sort == "random":
        logger.debug("shuffle {}".format(len(dataset.msgtext)))
        dataset.shuffle() # in time sequence is a good idea?
    elif args.sort == "time":
        logger.info("Sort by time...")
        dataset.sort(cmp=None, key=lambda x: time.strptime(x.time, '%Y-%m-%dT%H:%M:%SZ'),reverse=False)

    dataset.re_process_diff()
    logger.debug("re_process {}".format(len(dataset.msgtext)))

    dataset.deduplication()
    logger.debug("deduplicate {}".format(len(dataset.msgtext)))

    dataset.data_constraint(args.min_diff, args.max_diff, args.min_msg, args.max_msg)
    logger.debug("data_constraint {}".format(len(dataset.msgtext)))

    dataset.build_word2index2(output_folder)


    dataset.save_data(output_folder, True, True, True, True, True, True, True)
    logger.info("save {}".format(len(dataset.msgtext)))

    dataset.get_info()

    logger.info("The preprocessed data are stored in 11 files, {}/*.json".format(output_folder))
    time_end=time.time()
    logger.info('Time cost: {} min'.format((time_end-time_start)/60.0))
    
