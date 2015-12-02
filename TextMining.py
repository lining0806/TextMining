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


def MakeTextMining(*para):
    posts, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number, \
    lag, stopwords_set, blackwords_set, writewords_set, \
    all_words_tf_dict, all_words_df_dict, train_datas, test_speedup = para
    train_datas_count = len(train_datas)
    ## --------------------------------------------------------------------------------
    '''
    id_dict = {
        "NotPass":{
            id:None,
            ...
            },
        "Pass":{
            id:(tags,country,imp),
            ...
            }
    }
    '''
    id_dict = {
        "NotPass":{},
        "Pass":{}
    }
    ## --------------------------------------------------------------------------------
    ## 生成分类器模型
    feature_selection_flag = False
    my_selector = None
    if test_speedup and os.path.exists(fea_dict_file) and os.path.exists(best_clf_file):
        words_feature = []
        with open(fea_dict_file, 'r') as fp:
            for line in fp.readlines():
                word_feature = line.strip().decode("utf-8")
                words_feature.append(word_feature)
        if feature_selection_flag:
            with open(best_clf_file, "rb") as fp_pickle:
                my_selector, best_clf = pickle.load(fp_pickle)
        else:
            with open(best_clf_file, "rb") as fp_pickle:
                best_clf = pickle.load(fp_pickle)
    else:
        ## --------------------------------------------------------------------------------
        words_feature = MakeFeatureWordsDict(all_words_tf_dict, all_words_df_dict, stopwords_set, writewords_set, lag, fea_dict_size)
        train_features = []
        train_class = []
        for train_data in train_datas:
            TextFeatureClass = TextFeature(words_feature, train_data[0])
            train_features.append(TextFeatureClass.TextBool()) #### 可以调整特征抽取，训练集与测试集保持一致
            train_class.append(int(train_data[1])) # str转为int
        train_features = np.array(train_features)
        train_class = np.array(train_class)
        if feature_selection_flag:
            FeatureSelectorClass = FeatureSelector(train_features, train_class)
            my_selector, train_features = FeatureSelectorClass.PCA_Selector() #### 可以调整特征选择
        start_time_train = datetime.datetime.now()
        ClassifierTrainClass = ClassifierTrain(train_features, train_class)
        best_clf = ClassifierTrainClass.LR() #### 可以调整分类器训练
        end_time_train = datetime.datetime.now()
        print "best_clf training last time:", end_time_train-start_time_train
        if not os.path.exists(Classifier_Dir):
            os.makedirs(Classifier_Dir)
        with open(fea_dict_file, 'w') as fp:
            for word_feature in words_feature:
                fp.writelines(word_feature.encode("utf-8")) # 将unicode转换为utf-8
                fp.writelines("\n")
        if feature_selection_flag:
            with open(best_clf_file, "wb") as fp_pickle:
                pickle.dump((my_selector, best_clf), fp_pickle)
        else:
            with open(best_clf_file, "wb") as fp_pickle:
                pickle.dump(best_clf, fp_pickle)

    ## --------------------------------------------------------------------------------
    delta = datetime.timedelta(days=0, hours=8, minutes=0, seconds=0) # UTC刚好比CST晚8小时
    end_time = datetime.datetime.now()-delta
    start_time = end_time-datetime.timedelta(days=0, hours=0, minutes=30, seconds=0)-delta ## 可以修改查询的时间区段
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
            textseg_list = TextSeg(post[content_col], lag)
            textseg_set = set(textseg_list)
            ## --------------------------------------------------------------------------------
            #### 文本过滤
            if textseg_set & blackwords_set:
                print '{"_id":ObjectId("%s")} In Blackwords' % post["_id"]
                id_dict["NotPass"][post["_id"]] = None
            elif len(textseg_set)<=5 or len(textseg_list)<=8:
                print '{"_id":ObjectId("%s")} Too Short' % post["_id"]
                id_dict["NotPass"][post["_id"]] = None
            else:
                ## --------------------------------------------------------------------------------
                # ## 文本去重
                # if textseg_set not in id_dict["Pass"].values():
                #     id_dict["Pass"][post["_id"]] = textseg_set
                # else:
                #     print '{"_id":ObjectId("%s")} Duplicate' % post["_id"]
                #     id_dict["NotPass"][post["_id"]] = None
                ## --------------------------------------------------------------------------------
                ## 文本去重
                if id_dict["Pass"] == {}:
                    id_dict["Pass"][post["_id"]] = textseg_set
                else:
                    flag = 1
                    k_list = id_dict["Pass"].keys()
                    for k in k_list:
                        if id_dict["Pass"][k] & textseg_set == textseg_set: # 如果元素包含textseg_set，则不添加，包括二者相等情况
                            flag = 0
                            print '{"_id":ObjectId("%s")} Duplicate' % post["_id"]
                            id_dict["NotPass"][post["_id"]] = None
                            break
                        elif id_dict["Pass"][k] & textseg_set == id_dict["Pass"][k]: # 如果textseg_set包含元素，则除去元素添加textseg_set
                            id_dict["Pass"].pop(k)
                            print '{"_id":ObjectId("%s")} Duplicate' % k
                            id_dict["NotPass"][k] = None
                        else:
                            pass
                    if flag:
                        id_dict["Pass"][post["_id"]] = textseg_set
                ## --------------------------------------------------------------------------------
        else:
            print '{"_id":ObjectId("%s")} None' % post["_id"]
            id_dict["NotPass"][post["_id"]] = None
    ## --------------------------------------------------------------------------------
    len_pass, len_notpass = len(id_dict["Pass"]), len(id_dict["NotPass"])
    print "number", len_pass+len_notpass
    if len_pass+len_notpass>0:
        print "Pass Rate: %.2f%%" % (len_pass/(len_pass+len_notpass)*100)

    ## --------------------------------------------------------------------------------0
    for post in posts.find({"_id":{"$in":id_dict["Pass"].keys()}}):
        # print post[content_col]
        textseg_list = TextSeg(post[content_col], lag)
        textseg_set = set(textseg_list)
        ## --------------------------------------------------------------------------------
        #### 文本关键词提取
        TextExtractTagsClass = TextExtractTags(textseg_list, stopwords_set, writewords_set, topK=3)
        # tags = TextExtractTagsClass.Tags_Words_Feature(words_feature)
        # tags = TextExtractTagsClass.Tags_Tf(lag)
        tags = TextExtractTagsClass.Tags_TfIDf(all_words_df_dict, train_datas_count, lag)
        print '{"_id":ObjectId("%s")} ' % post["_id"],
        for tag in tags:
            print tag,
        print ""
        ## --------------------------------------------------------------------------------
        #### 文本分类
        TextFeatureClass = TextFeature(words_feature, textseg_list)
        test_features = TextFeatureClass.TextBool() #### 可以调整特征抽取，训练集与测试集保持一致
        test_features = np.array(test_features)
        '''
        Reshape your data
        either using X.reshape(-1, 1) if your data has a single feature
        or X.reshape(1, -1) if it contains a single sample.
        '''
        test_features = test_features.reshape(1, -1)
        if feature_selection_flag:
            test_features = my_selector.transform(test_features)
        test_class = best_clf.predict(test_features)
        print '{"_id":ObjectId("%s")} ' % post["_id"], Number_Country_Map[str(test_class[0])] # int转为str
        ## --------------------------------------------------------------------------------
        #### 文本推荐
        level = "1"
        if datetime.time(0, 0, 0)<post[time_col].time()<datetime.time(6, 0, 0):
            level = "2"
            digits = [word for word in textseg_list if word.isdigit()]
            if len(textseg_set & writewords_set)>=1 and len(digits)>=3 and len(textseg_set)>=10 and len(textseg_list)>=16:
                level = "3"
        print '{"_id":ObjectId("%s")} ' % post["_id"], level
        ## --------------------------------------------------------------------------------
        id_dict["Pass"][post["_id"]] = (tags, Number_Country_Map[str(test_class[0])], level)

    ## --------------------------------------------------------------------------------
    return id_dict

def MakeTextMining_ClassifyTest(*para):
    posts, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, limit_number, \
    lag, stopwords_set, blackwords_set, writewords_set, \
    all_words_tf_dict, all_words_df_dict, train_datas, test_speedup = para
    ## --------------------------------------------------------------------------------
    '''
    id_dict = {
        "NotPass":{
            id:None,
            ...
            },
        "Pass":{
            id:(tags,country,imp),
            ...
            }
    }
    '''
    id_dict = {
        "NotPass":{},
        "Pass":{}
    }
    ## --------------------------------------------------------------------------------
    ## 生成分类器模型
    feature_selection_flag = False
    my_selector = None
    if test_speedup and os.path.exists(fea_dict_file) and os.path.exists(best_clf_file):
        words_feature = []
        with open(fea_dict_file, 'r') as fp:
            for line in fp.readlines():
                word_feature = line.strip().decode("utf-8")
                words_feature.append(word_feature)
        if feature_selection_flag:
            with open(best_clf_file, "rb") as fp_pickle:
                my_selector, best_clf = pickle.load(fp_pickle)
        else:
            with open(best_clf_file, "rb") as fp_pickle:
                best_clf = pickle.load(fp_pickle)
    else:
        ## --------------------------------------------------------------------------------
        words_feature = MakeFeatureWordsDict(all_words_tf_dict, all_words_df_dict, stopwords_set, writewords_set, lag, fea_dict_size)
        train_features = []
        train_class = []
        for train_data in train_datas:
            TextFeatureClass = TextFeature(words_feature, train_data[0])
            train_features.append(TextFeatureClass.TextBool()) #### 可以调整特征抽取，训练集与测试集保持一致
            train_class.append(int(train_data[1])) # str转为int
        train_features = np.array(train_features)
        train_class = np.array(train_class)
        if feature_selection_flag:
            FeatureSelectorClass = FeatureSelector(train_features, train_class)
            my_selector, train_features = FeatureSelectorClass.PCA_Selector() #### 可以调整特征选择
        start_time_train = datetime.datetime.now()
        ClassifierTrainClass = ClassifierTrain(train_features, train_class)
        best_clf = ClassifierTrainClass.LR() #### 可以调整分类器训练
        end_time_train = datetime.datetime.now()
        print "best_clf training last time:", end_time_train-start_time_train
        if not os.path.exists(Classifier_Dir):
            os.makedirs(Classifier_Dir)
        with open(fea_dict_file, 'w') as fp:
            for word_feature in words_feature:
                fp.writelines(word_feature.encode("utf-8")) # 将unicode转换为utf-8
                fp.writelines("\n")
        if feature_selection_flag:
            with open(best_clf_file, "wb") as fp_pickle:
                pickle.dump((my_selector, best_clf), fp_pickle)
        else:
            with open(best_clf_file, "wb") as fp_pickle:
                pickle.dump(best_clf, fp_pickle)

    ## --------------------------------------------------------------------------------
    start_time = datetime.datetime(2014, 1, 1)
    end_time = datetime.datetime.now()
    count = 0
    correct_count = 0
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
            textseg_list = TextSeg(post[content_col], lag)
            count += 1
            ## --------------------------------------------------------------------------------
            #### 文本分类
            TextFeatureClass = TextFeature(words_feature, textseg_list)
            test_features = TextFeatureClass.TextBool() #### 可以调整特征抽取，训练集与测试集保持一致
            test_features = np.array(test_features)
            '''
            Reshape your data
            either using X.reshape(-1, 1) if your data has a single feature
            or X.reshape(1, -1) if it contains a single sample.
            '''
            test_features = test_features.reshape(1, -1)
            test_class = best_clf.predict(test_features)
            if Number_Country_Map[str(test_class[0])] == post[country_col]:
                correct_count += 1
            print '{"_id":ObjectId("%s")} ' % post["_id"], Number_Country_Map[str(test_class[0])] # int转为str
        else:
            print '{"_id":ObjectId("%s")} None' % post["_id"]
    print "number of all the train datas:", count
    print "all correct classification data number:", correct_count
    if count>0:
        print "accuracy of classification: %.2f%%" % (correct_count/count*100)

    ## --------------------------------------------------------------------------------
    return id_dict
