# coding: utf-8
from __future__ import division
__author__ = 'LiNing'


import os
import shutil
import re
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
        # if not sorted_word.isdigit() and not sorted_word == "\r\n": # 不是数字
        # if not sorted_word.isdigit() and not sorted_word == "\r\n" and not sorted_word in filterword_set: # 不是数字
        # if re.match(ur'^[a-z A-Z -]+$', sorted_word) and not sorted_word == "\r\n": # 英文
        # if re.match(ur'^[a-z A-Z -]+$', sorted_word) and not sorted_word == "\r\n" and not sorted_word in filterword_set: # 英文
        # if re.match(ur'^[\u4e00-\u9fa5]+$', sorted_word) and not sorted_word == "\r\n": # 中文
        # if re.match(ur'^[\u4e00-\u9fa5]+$', sorted_word) and not sorted_word == "\r\n" and not sorted_word in filterword_set: # 中文
        if re.match(ur'^[\u4e00-\u9fa5]+$|^[a-z A-Z -]+$', sorted_word) and not sorted_word == "\r\n" and not sorted_word in filterword_set: # 中英文
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
    all_words_dict = {}
    all_source_dict = {}
    all_country_dict = {}
    ## --------------------------------------------------------------------------------
    start_time = datetime.datetime(2014, 1, 1)
    end_time = datetime.datetime.now()
    count = 0
    train_datas = []
    for post in posts.find({  ##################################### 查询操作
        time_col:{"$gte":start_time, "$lte":end_time},
        content_col:{"$exists":1},
        source_col:{"$exists":1},
        t_status_col:1, # 已发布的
        keyword_col:{"$exists":1}, country_col:{"$exists":1}, imp_col:{"$exists":1},
    },): #.sort(time_col, pymongo.DESCENDING).limit(limit_number):
        ## --------------------------------------------------------------------------------
        # print post
        if post[content_col] is not None:
            # print post[content_col]
            count += 1
            ## --------------------------------------------------------------------------------
            textseg_list = TextSeg(post[content_col], lag)
            for testseg in textseg_list:
                if all_words_dict.has_key(testseg):
                    all_words_dict[testseg] += 1
                else:
                    all_words_dict[testseg] = 1
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
            ## --------------------------------------------------------------------------------
            if post[country_col] in Country_Number_Map:
                train_datas.append((textseg_list, Country_Number_Map[post[country_col]]))
        else:
            print '{"_id":ObjectId("%s")} None' % post["_id"]
    print "all true data number:", count
    ## --------------------------------------------------------------------------------
    return all_words_dict, all_source_dict, all_country_dict, train_datas

def MakeTextMining(*para):
    posts, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number, \
    lag, stopwords_set, notstopwords_set, words_feature, train_datas, test_speedup = para
    ## --------------------------------------------------------------------------------
    '''
    id_dict = {
        "Pass":{
            id:content,
            ...
            },
        "NotPass":{
            id:(tags,country,imp),
            ...
            }
    }
    '''
    id_dict = {
        "Pass":{},
        "NotPass":{}
    }
    ## --------------------------------------------------------------------------------
    ## 生成最优分类器
    if test_speedup and os.path.exists(best_clf_file):
        with open(best_clf_file, "rb") as fp_pickle:
            best_clf = pickle.load(fp_pickle)
    else:
        train_features = []
        train_class = []
        for train_data in train_datas:
            TextFeatureClass = TextFeature(words_feature, train_data[0])
            train_features.append(TextFeatureClass.TextBool()) #### 可以调整特征抽取
            train_class.append(int(train_data[1])) # str转为int
        train_features = np.array(train_features)
        train_class = np.array(train_class)
        start_time_train = datetime.datetime.now()
        ClassifierTrainClass = ClassifierTrain(train_features, train_class)
        best_clf = ClassifierTrainClass.LibSVM() #### 可以调整分类器训练
        end_time_train = datetime.datetime.now()
        print "best_clf training last time:", end_time_train-start_time_train
        if not os.path.exists(Model_Dir):
            os.makedirs(Model_Dir)
        with open(best_clf_file, "wb") as fp_pickle:
            pickle.dump(best_clf, fp_pickle)

    ## --------------------------------------------------------------------------------
    end_time = datetime.datetime.now()
    start_time = end_time-datetime.timedelta(days=1, hours=0, minutes=0, seconds=0) ## 可以修改查询的时间区段
    for post in posts.find({  ##################################### 查询操作
        time_col:{"$gte":start_time, "$lte":end_time},
        content_col:{"$exists":1},
        source_col:{"$exists":1},
        t_status_col:0, # 未发布的
        keyword_col:{"$exists":0}, country_col:{"$exists":0}, imp_col:{"$exists":0},
    },).sort(time_col, -1).limit(limit_number):
        ## --------------------------------------------------------------------------------
        # print post
        if post[content_col] is not None:
            # print post[content_col]
            ## --------------------------------------------------------------------------------
            textseg_list = TextSeg(post[content_col], lag)
            textseg_set = set(textseg_list)
            if stopwords_set & textseg_set or len(textseg_set)<=3 or len(textseg_list)<=5: #### 文本过滤
                print '{"_id":ObjectId("%s")} In Stopwords or Too Short' % post["_id"]
                id_dict["NotPass"][post["_id"]] = post[content_col]
            else:
                ## --------------------------------------------------------------------------------
                #### 文本关键词提取
                tags = TextExtractTags(words_feature, textseg_list, notstopwords_set, topK=3)
                print '{"_id":ObjectId("%s")} ' % post["_id"],
                for tag in tags:
                    print tag,
                print ""
                ## --------------------------------------------------------------------------------
                #### 文本分类
                TextFeatureClass = TextFeature(words_feature, textseg_list)
                test_features = TextFeatureClass.TextBool() #### 可以调整特征抽取
                test_features = np.array(test_features)
                test_class = ClassifierTest(best_clf, test_features)
                print '{"_id":ObjectId("%s")} ' % post["_id"], Number_Country_Map[str(test_class[0])] # int转为str
                ## --------------------------------------------------------------------------------
                #### 文本推荐
                level = "1"
                if datetime.time(0, 0, 0)<post[time_col].time()<datetime.time(6, 0, 0):
                    level = "2"
                    digits = [word for word in textseg_list if word.isdigit()]
                    if len(notstopwords_set & textseg_set)>3 and len(digits)>3:
                        level = "3"
                print '{"_id":ObjectId("%s")} ' % post["_id"], level
                ## --------------------------------------------------------------------------------
                id_dict["Pass"][post["_id"]] = (tags, Number_Country_Map[str(test_class[0])], level)
        else:
            print '{"_id":ObjectId("%s")} None' % post["_id"]
            id_dict["NotPass"][post["_id"]] = post[content_col]
    ## --------------------------------------------------------------------------------
    len_pass, len_notpass = len(id_dict["Pass"]), len(id_dict["NotPass"])
    if len_pass+len_notpass>0:
        print "Pass Rate: %.2f%%" % (len_pass/(len_pass+len_notpass)*100)
    return id_dict

def DataSendMail(*para):
    smtp_server, from_addr, passwd, to_addr, from_addr, sendmail_flag, id_dict = para
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
    ## 一旦训练集变化，则需要事先手动删除all_sorted_words_list和train_datas
    all_sorted_words_list = []
    if os.path.exists(all_sorted_words_list_file) and os.path.exists(train_datas_file):
        with open(all_sorted_words_list_file, 'r') as fp:
            for line in fp.readlines():
                sorted_word = line.strip().decode("utf-8")
                all_sorted_words_list.append(sorted_word)
        # with open(train_datas_file, "rb") as fp_pickle:
        #     train_datas = pickle.load(fp_pickle)
        with open(train_datas_file, "rb") as fp_json:
            train_datas = json.load(fp_json)
    else:
        if not os.path.exists(Datas_Dir):
            os.makedirs(Datas_Dir)
        all_para = (posts,
                    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number,
                    lag)
        all_words_dict, all_source_dict, all_country_dict, train_datas = MakeAllWordsDict(*all_para)
        all_sorted_words_tuple = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True)
        with open(all_sorted_words_list_file, 'w') as fp:
            for sorted_word, times in all_sorted_words_tuple:
                # print sorted_word, times
                all_sorted_words_list.append(sorted_word)
                fp.writelines(sorted_word.encode("utf-8")) # 将unicode转换为utf-8
                fp.writelines("\n")
        # with open(train_datas_file, "wb") as fp_pickle:
        #     pickle.dump(train_datas, fp_pickle)
        with open(train_datas_file, "wb") as fp_json:
            json.dump(train_datas, fp_json)
        all_sorted_source_tuple = sorted(all_source_dict.items(), key=lambda f:f[1], reverse=True)
        for sorted_source, times in all_sorted_source_tuple:
            print sorted_source, times
        all_sorted_country_tuple = sorted(all_country_dict.items(), key=lambda f:f[1], reverse=True)
        for sorted_country, times in all_sorted_country_tuple:
            print sorted_country, times
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
                lag, stopwords_set, notstopwords_set, words_feature, train_datas, test_speedup)
    id_dict = MakeTextMining(*all_para)

    all_para = (smtp_server, from_addr, passwd, to_addr, from_addr, sendmail_flag, id_dict)
    DataSendMail(*all_para)

    # all_para = (posts, keyword_col, country_col, imp_col, id_dict)
    # DataUpdateAndSave(*all_para)

    ## --------------------------------------------------------------------------------
