# coding: utf-8
from __future__ import division
__author__ = 'LiNing'


import os
import shutil
import re
import math
import nltk
import jieba
import jieba.analyse
import pymongo
import datetime
import numpy as np
try:
   import cPickle as pickle
except ImportError:
   import pickle
try:
   import simplejson as json
except ImportError:
   import json

from TextConfig import *
from TextProcess import *
from SendMail import *
from TextMining import *


def MakeFeatureWordsDict(all_sorted_words_list, stopwords_set, notstopwords_set, lag, fea_dict_size): # 特征词words_feature是选用的word-词典
    ## --------------------------------------------------------------------------------
    words_feature = list(notstopwords_set)
    n = len(words_feature)
    assert(n<=fea_dict_size)
    filterword_set = stopwords_set | notstopwords_set
    if lag == "eng": # 英文情况
        wordlen_min, wordlen_max = 2, 15
    elif lag == "chs": # 中文情况
        wordlen_min, wordlen_max = 1, 5
    else:
        wordlen_min, wordlen_max = 1, 15
    for sorted_word in all_sorted_words_list:
        if n == fea_dict_size:
            break
        # if not sorted_word.isdigit(): # 不是数字
        if re.match(ur'^[\u4e00-\u9fa5]+$|^[a-z A-Z -]+$', sorted_word) and not sorted_word in filterword_set: # 中英文
            if wordlen_min<len(sorted_word)<wordlen_max: # unicode长度
                words_feature.append(sorted_word)
                n += 1
    print "all_words length in words_feature: ", len(words_feature)
    # for word_feature in words_feature:
    #     print word_feature
    return words_feature

def MakeAllWordsDict(*para):
    posts, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number, \
    lag = para
    ## --------------------------------------------------------------------------------
    all_words_tf_dict = {}
    all_words_df_dict = {}
    train_datas = []
    all_source_dict = {}
    all_country_dict = {}
    ## --------------------------------------------------------------------------------
    start_time = datetime.datetime(2014, 1, 1)
    end_time = datetime.datetime.now()
    for post in posts.find({  ##################################### 查询操作
        time_col:{"$gte":start_time, "$lte":end_time},
        content_col:{"$exists":1},
        source_col:{"$exists":1},
        t_status_col:1, # 已发布的
        keyword_col:{"$exists":1}, country_col:{"$exists":1}, imp_col:{"$exists":1},
    },): #.sort(time_col, pymongo.DESCENDING).limit(limit_number):
        ## --------------------------------------------------------------------------------
        # print post
        if post[content_col] is not None and post[country_col] in Country_Number_Map:
            # print post[content_col]
            ## --------------------------------------------------------------------------------
            textseg_list = TextSeg(post[content_col], lag)
            for testseg in textseg_list: ## all_words_tf_dict为该词的词频数
                if all_words_tf_dict.has_key(testseg):
                    all_words_tf_dict[testseg] += 1
                else:
                    all_words_tf_dict[testseg] = 1
            for testseg in set(textseg_list): ## all_words_df_dict为包含该词的文档频率
                if all_words_df_dict.has_key(testseg):
                    all_words_df_dict[testseg] += 1
                else:
                    all_words_df_dict[testseg] = 1
            train_datas.append((textseg_list, Country_Number_Map[post[country_col]]))
            ## --------------------------------------------------------------------------------
            if all_source_dict.has_key(post[source_col]):
                all_source_dict[post[source_col]] += 1
            else:
                all_source_dict[post[source_col]] = 1
            ## --------------------------------------------------------------------------------
            if all_country_dict.has_key(post[country_col]):
                all_country_dict[post[country_col]] += 1
            else:
                all_country_dict[post[country_col]] = 1
        else:
            print '{"_id":ObjectId("%s")} None or Not in Map' % post["_id"]
    ## --------------------------------------------------------------------------------
    train_datas_count = len(train_datas)
    print "number of all the train datas:", train_datas_count
    for k in all_words_df_dict:
        all_words_df_dict[k] = math.log(train_datas_count/(1+all_words_df_dict[k]))
    ## --------------------------------------------------------------------------------
    return all_words_tf_dict, all_words_df_dict, train_datas, all_source_dict, all_country_dict

def DataSendMail(*para):
    smtp_server, from_addr, passwd, to_addr, sendmail_flag, id_dict = para
    text = ''
    for id_data in id_dict["NotPass"]:
        id_content = '{"_id":ObjectId("%s")} In Stopwords\n%s\n' % (id_data, id_dict["NotPass"][id_data])
        text += id_content
        # text = text.join(id_content) # 错误！join是循环用text连接id_content的字符
    if sendmail_flag and text != '':
        subject = 'Waring...'
        files = []
        send_mail(smtp_server, from_addr, passwd, to_addr, subject, text, files)

def DataUpdateAndSave(*para):
    posts, keyword_col, country_col, imp_col, id_dict = para
    for id_data in id_dict["Pass"]:
        posts.update({"_id":id_data}, {"$set":{keyword_col:id_dict["Pass"][id_data][0],
                                               country_col:id_dict["Pass"][id_data][1],
                                               imp_col:id_dict["Pass"][id_data][2],
                                               "t_status":1}})
        ## --------------------------------------------------------------------------------
        print '{"_id":ObjectId("%s")} Update' % id_data


if __name__ == '__main__':
    ## --------------------------------------------------------------------------------
    ## 一旦训练集变化，则需要事先手动删除all_words_dict_file和train_datas
    if os.path.exists(all_words_dict_file) and os.path.exists(train_datas_file):
        # with open(all_words_dict_file, "rb") as fp_pickle:
        #     all_words_tf_dict, all_words_df_dict = pickle.load(fp_pickle)
        # with open(train_datas_file, "rb") as fp_pickle:
        #     train_datas = pickle.load(fp_pickle)
        with open(all_words_dict_file, "rb") as fp_json:
            all_words_tf_dict, all_words_df_dict = json.load(fp_json)
        with open(train_datas_file, "rb") as fp_json:
            train_datas = json.load(fp_json)
    else:
        if not os.path.exists(Datas_Dir):
            os.makedirs(Datas_Dir)
        all_para = (posts,
                    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number,
                    lag)
        all_words_tf_dict, all_words_df_dict, train_datas, all_source_dict, all_country_dict = MakeAllWordsDict(*all_para)
        # with open(all_words_dict_file, "wb") as fp_pickle:
        #     pickle.dump((all_words_tf_dict, all_words_df_dict), fp_pickle)
        # with open(train_datas_file, "wb") as fp_pickle:
        #     pickle.dump(train_datas, fp_pickle)
        with open(all_words_dict_file, "wb") as fp_json:
            json.dump((all_words_tf_dict, all_words_df_dict), fp_json)
        with open(train_datas_file, "wb") as fp_json:
            json.dump(train_datas, fp_json)
        for key_source in all_source_dict:
            print key_source, all_source_dict[key_source]
        for key_country in all_country_dict:
            print key_country, all_country_dict[key_country]
        #### 删除之前与之相关的文件
        if os.path.exists(Fea_Dict_Dir):
            shutil.rmtree(Fea_Dict_Dir)
        if os.path.exists(Model_Dir):
            shutil.rmtree(Model_Dir)

    ## --------------------------------------------------------------------------------
    ## 生成特征词
    fea_dict_file = os.path.join(Fea_Dict_Dir, "fea_dict_%d" % fea_dict_size)
    if test_speedup and os.path.exists(fea_dict_file):
        words_feature = []
        with open(fea_dict_file, 'r') as fp:
            for line in fp.readlines():
                word_feature = line.strip().decode("utf-8")
                words_feature.append(word_feature)
    else:
        ## --------------------------------------------------------------------------------
        all_sorted_words_tuple_list = sorted(all_words_tf_dict.items(), key=lambda f:f[1], reverse=True)
        # all_sorted_words_tuple_list = sorted(all_words_df_dict.items(), key=lambda f:f[1], reverse=True)
        ## --------------------------------------------------------------------------------
        all_sorted_words_list = list(zip(*all_sorted_words_tuple_list)[0])
        # all_sorted_words_list = []
        # for sorted_word, times in all_sorted_words_tuple_list:
        #     all_sorted_words_list.append(sorted_word)
        words_feature = MakeFeatureWordsDict(all_sorted_words_list, stopwords_set, notstopwords_set, lag, fea_dict_size)
        if not os.path.exists(Fea_Dict_Dir):
            os.makedirs(Fea_Dict_Dir)
        with open(fea_dict_file, 'w') as fp:
            for word_feature in words_feature:
                fp.writelines(word_feature.encode("utf-8")) # 将unicode转换为utf-8
                fp.writelines("\n")

    ## --------------------------------------------------------------------------------
    all_para = (posts,
                time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number,
                lag, stopwords_set, notstopwords_set, words_feature, all_words_df_dict, train_datas, test_speedup)
    id_dict = MakeTextMining(*all_para)
    # id_dict = MakeTextMining_ClassifyTest(*all_para) # 测试不同分类器性能

    all_para = (smtp_server, from_addr, passwd, to_addr, sendmail_flag, id_dict)
    DataSendMail(*all_para)

    # all_para = (posts, keyword_col, country_col, imp_col, id_dict)
    # DataUpdateAndSave(*all_para)

    ## --------------------------------------------------------------------------------
