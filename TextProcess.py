#coding: utf-8
from __future__ import division
__author__ = 'LiNing'


import re
import math
import nltk
import jieba
import jieba.analyse
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def TextSeg(text, lag):
    if lag == "eng": # 英文情况
        word_list = nltk.word_tokenize(text)
    elif lag == "chs": # 中文情况
        ## --------------------------------------------------------------------------------
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        word_cut = jieba.cut(text, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut) # genertor转化为list，每个词unicode格式
        # jieba.disable_parallel() # 关闭并行分词模式
        ## --------------------------------------------------------------------------------
        # # jieba关键词提取
        # tags = jieba.analyse.extract_tags(text, topK=10)
        # # tags = jieba.analyse.textrank(text, topK=10)
        # print tags
        ## --------------------------------------------------------------------------------
    else:
        word_list = []
    return word_list

def WordLen(lag):
    if lag == "eng": # 英文情况
        return 2, 15
    elif lag == "chs": # 中文情况
        return 1, 5
    else:
        return 1, 15

def MakeFeatureWordsDict(all_words_tf_dict, all_words_df_dict, stopwords_set, writewords_set, lag, fea_dict_size):
    ## --------------------------------------------------------------------------------
    words_feature = list(writewords_set)
    n = len(words_feature)
    assert(n<=fea_dict_size)
    filterword_set = stopwords_set | writewords_set
    wordlen_min, wordlen_max = WordLen(lag)
    all_sorted_words_tuple_list = sorted(all_words_tf_dict.items(), key=lambda f:f[1], reverse=True)
    # all_sorted_words_tuple_list = sorted(all_words_df_dict.items(), key=lambda f:f[1], reverse=True)
    all_sorted_words_list = list(zip(*all_sorted_words_tuple_list)[0])
    # all_sorted_words_list = []
    # for sorted_word, times in all_sorted_words_tuple_list:
    #     all_sorted_words_list.append(sorted_word)
    ## --------------------------------------------------------------------------------
    for sorted_word in all_sorted_words_list:
        if n == fea_dict_size:
            break
        # if not sorted_word.isdigit(): # 不是数字
        if re.match(ur'^[\u4e00-\u9fa5]+$|^[a-z A-Z -]+$', sorted_word) and sorted_word not in filterword_set: # 中英文
            if wordlen_min<len(sorted_word)<wordlen_max: # unicode长度
                words_feature.append(sorted_word)
                n += 1
    print "all_words length in words_feature: ", len(words_feature)
    # for word_feature in words_feature:
    #     print word_feature
    return words_feature

class TextExtractTags(object):
    # 申明相关的属性
    def __init__(self, text, stopwords_set, writewords_set, topK=10):
        self.text = text
        self.stopwords_set = stopwords_set
        self.writewords_set = writewords_set
        self.topK = topK

    def SelectK(self, words_dict):
        ## --------------------------------------------------------------------------------
        words_tuple_list = sorted(words_dict.items(), key=lambda f:f[1], reverse=True)
        sorted_words = list(zip(*words_tuple_list)[0])
        # sorted_words = []
        # for key, value in words_tuple_list:
        #     sorted_words.append(key)
        #### 直接截断
        # new_sorted_words = filter(lambda f:f not in self.stopwords_set, sorted_words)
        ####
        new_sorted_words = filter(lambda f:f in self.writewords_set, sorted_words)
        new_sorted_words.extend(filter(lambda f:f not in (self.stopwords_set | self.writewords_set), sorted_words))
        ## --------------------------------------------------------------------------------
        tags = new_sorted_words[:self.topK]
        return tags

    def Tags_Words_Feature(self, words_feature):
        ## --------------------------------------------------------------------------------
        tf_dict = {}
        for word in self.text:
            if tf_dict.has_key(word):
                tf_dict[word] += 1
            else:
                if word in words_feature:
                    tf_dict[word] = 1
        length = len(self.text)
        for key in tf_dict:
            tf_dict[key] /= length
        return self.SelectK(tf_dict)

    def Tags_Tf(self, lag):
        ## --------------------------------------------------------------------------------
        wordlen_min, wordlen_max = WordLen(lag)
        tf_dict = {}
        for word in self.text:
            if tf_dict.has_key(word):
                tf_dict[word] += 1
            else:
                if re.match(ur'^[\u4e00-\u9fa5]+$|^[a-z A-Z -]+$', word) and wordlen_min<len(word)<wordlen_max:
                    tf_dict[word] = 1
        length = len(self.text)
        for key in tf_dict:
            tf_dict[key] /= length
        return self.SelectK(tf_dict)

    def Tags_TfIDf(self, all_words_df_dict, train_datas_count, lag):
        ## --------------------------------------------------------------------------------
        wordlen_min, wordlen_max = WordLen(lag)
        tf_idf_dict = {}
        for word in self.text:
            if tf_idf_dict.has_key(word):
                tf_idf_dict[word] += 1
            else:
                if re.match(ur'^[\u4e00-\u9fa5]+$|^[a-z A-Z -]+$', word) and wordlen_min<len(word)<wordlen_max:
                    tf_idf_dict[word] = 1
        length = len(self.text)
        for key in tf_idf_dict:
            if key in all_words_df_dict:
                tf_idf_dict[key] = tf_idf_dict[key]/length*all_words_df_dict[key]
            else:
                tf_idf_dict[key] = tf_idf_dict[key]/length*math.log(train_datas_count)
        return self.SelectK(tf_idf_dict)

class TextFeature(object):
    # 申明相关的属性
    def __init__(self, words_feature, text):
        self.words_feature = words_feature
        self.text = text

    def TextBool(self):
        bool_features = []
        words = set(self.text)
        for word_feature in self.words_feature:
            if word_feature in words:
                bool_features.append(1)
            else:
                bool_features.append(0)
        return bool_features

    def TextTf(self):
        tf_features = []
        length = len(self.text)
        for word_feature in self.words_feature:
            word_count = self.text.count(word_feature)
            tf = word_count/length
            tf_features.append(tf)
        return tf_features

    def TextIDf(self, all_words_df_dict):
        idf_features = []
        for word_feature in self.words_feature:
            idf = all_words_df_dict[word_feature]
            idf_features.append(idf)
        return idf_features

    def TextTfIDf(self, all_words_df_dict):
        tf_idf_features = []
        length = len(self.text)
        for word_feature in self.words_feature:
            word_count = self.text.count(word_feature)
            tf = word_count/length
            idf = all_words_df_dict[word_feature]
            tf_idf = tf*idf
            tf_idf_features.append(tf_idf)
        return tf_idf_features

class FeatureSelector(object):
    # 申明相关的属性
    def __init__(self, train_features, train_class, k=1000):
        self.train_features = train_features
        self.train_class = train_class
        self.k = k

    def PCA_Selector(self):
        my_selector = PCA(n_components=self.k).fit(self.train_features)
        train_features = my_selector.transform(self.train_features)
        return my_selector, train_features

    def KBest_Selector(self):
        my_selector = SelectKBest(score_func=f_classif, k=self.k).fit(self.train_features, self.train_class)
        train_features = my_selector.transform(self.train_features)
        return my_selector, train_features

class ClassifierTrain(object):
    # 申明相关的属性
    def __init__(self, train_features, train_class):
        self.train_features = train_features
        self.train_class = train_class

    # def SVM(self):
    #     ## Pipeline+GridSearchCV
    #     parameters = [
    #         {
    #             'pca__n_components':[10, 15, 20, 25, 30],
    #             'svm__kernel':['rbf'],
    #             'svm__gamma':[1e-3, 1e-2, 1e-1],
    #             'svm__C':[1e-2, 1e-1, 1, 5, 10]
    #         },
    #         {
    #             'pca__n_components':[10, 15, 20, 25, 30],
    #             'svm__kernel':['linear'],
    #             'svm__C':[1e-2, 1e-1, 1, 5, 10]
    #         }
    #     ]
    #     # parameters = {
    #     #     'pca__n_components':[10, 15, 20, 25, 30],
    #     #     'svm__kernel':['rbf'],
    #     #     'svm__gamma':[1e-3, 1e-2, 1e-1],
    #     #     'svm__C':[1e-2, 1e-1, 1, 5, 10]
    #     # }
    #     # print list(ParameterGrid(parameters))
    #     pipeline = Pipeline(
    #         steps = [
    #             ('pca', PCA()), # 'pca'对应'pca__'
    #             ('svm', SVC()) # 'svm'对应'svm__'
    #         ]
    #     )
    #     clf = GridSearchCV(
    #         estimator = pipeline,
    #         param_grid = parameters,
    #         cv = StratifiedKFold(self.train_class, 5),
    #         scoring = "accuracy",
    #         n_jobs = 3
    #     )
    #     clf.fit(self.train_features, self.train_class)
    #     best_clf = clf.best_estimator_
    #     return best_clf

    def SVM(self):
        clf = SVC()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

    def LibSVM(self):
        clf = LinearSVC()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

    def LR(self):
        clf = LogisticRegression()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

    def NB(self):
        clf = MultinomialNB()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

    def DT(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

    def KNN(self):
        clf = KNeighborsClassifier(n_neighbors=100)
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf
