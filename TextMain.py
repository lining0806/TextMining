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


def MakeAllWordsDict(*para):
    posts, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number, \
    lag = para
    ## --------------------------------------------------------------------------------
    all_words_tf_dict = {}
    all_words_df_dict = {}
    train_datas = []
    # all_source_dict = {}
    # all_country_dict = {}
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
            textseg_list = TextSeg(post[content_col], lag)
            ## --------------------------------------------------------------------------------
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
            # if all_source_dict.has_key(post[source_col]):
            #     all_source_dict[post[source_col]] += 1
            # else:
            #     all_source_dict[post[source_col]] = 1
            # if all_country_dict.has_key(post[country_col]):
            #     all_country_dict[post[country_col]] += 1
            # else:
            #     all_country_dict[post[country_col]] = 1
        else:
            print '{"_id":ObjectId("%s")} None or Not in Map' % post["_id"]
    ## --------------------------------------------------------------------------------
    # for key_source in all_source_dict:
    #     print key_source, all_source_dict[key_source]
    # for key_country in all_country_dict:
    #     print key_country, all_country_dict[key_country]
    train_datas_count = len(train_datas)
    print "number of all the train datas:", train_datas_count
    for k in all_words_df_dict:
        all_words_df_dict[k] = math.log(train_datas_count/(1+all_words_df_dict[k]))
    ## --------------------------------------------------------------------------------
    return all_words_tf_dict, all_words_df_dict, train_datas

def DataSendMail(*para):
    smtp_server, from_addr, passwd, to_addr, sendmail_flag, id_dict = para
    text = ''
    for id_data in id_dict["NotPass"]:
        id_content = '{"_id":ObjectId("%s")} NotPass\n%s\n' % (id_data, id_dict["NotPass"][id_data])
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
    ## 一旦训练集变化，则需要事先手动删除all_words_dict_file和train_datas_file
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
        all_words_tf_dict, all_words_df_dict, train_datas = MakeAllWordsDict(*all_para)
        # with open(all_words_dict_file, "wb") as fp_pickle:
        #     pickle.dump((all_words_tf_dict, all_words_df_dict), fp_pickle)
        # with open(train_datas_file, "wb") as fp_pickle:
        #     pickle.dump(train_datas, fp_pickle)
        with open(all_words_dict_file, "wb") as fp_json:
            json.dump((all_words_tf_dict, all_words_df_dict), fp_json)
        with open(train_datas_file, "wb") as fp_json:
            json.dump(train_datas, fp_json)
        #### 删除之前与之相关的文件
        if os.path.exists(Classifier_Dir):
            shutil.rmtree(Classifier_Dir)

    ## --------------------------------------------------------------------------------
    all_para = (posts,
                time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number,
                lag, stopwords_set, blackwords_set, writewords_set,
                all_words_tf_dict, all_words_df_dict, train_datas, test_speedup)
    id_dict = MakeTextMining(*all_para)
    # MakeTextMining_ClassifyTest(*all_para) # 测试不同分类器性能
    # MakeTextMining_Calendar(*all_para) # 财经日历

    ## --------------------------------------------------------------------------------
    # all_para = (smtp_server, from_addr, passwd, to_addr, sendmail_flag, id_dict)
    # DataSendMail(*all_para)

    # all_para = (posts, keyword_col, country_col, imp_col, id_dict)
    # DataUpdateAndSave(*all_para)

    ## --------------------------------------------------------------------------------
