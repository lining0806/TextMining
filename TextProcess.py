#coding: utf-8
from __future__ import division
__author__ = 'LiNing'

import nltk
import jieba
import jieba.analyse
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
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

def TextExtractTags(words_feature, text, notstopwords_set, topK=10): # 每个text对应tags
    ## --------------------------------------------------------------------------------
    tf_dict = {}
    for word in text:
        if tf_dict.has_key(word):
            tf_dict[word] += 1
        else:
            if word in words_feature:
                tf_dict[word] = 1
    length = len(text)
    for key in tf_dict:
        tf_dict[key] /= length
    ## --------------------------------------------------------------------------------
    # tf_dict = {}
    # for word_feature in words_feature:
    #     word_count = text.count(word_feature)
    #     length = len(text)
    #     tf = word_count/length
    #     tf_dict[word_feature] = tf
    ## --------------------------------------------------------------------------------
    # key函数利用词频进行降序排序
    tf_list = sorted(tf_dict.items(), key=lambda f:f[1], reverse=True)
    sorted_words = []
    for key, value in tf_list:
        sorted_words.append(key)
    #### 直接截断
    # new_sorted_words = sorted_words
    #### 参考非停用词进行调序
    new_sorted_words = filter(lambda f:f in notstopwords_set, sorted_words)
    new_sorted_words.extend(filter(lambda f:f not in notstopwords_set, sorted_words))
    ## --------------------------------------------------------------------------------
    tags = new_sorted_words[:topK]
    return tags

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


class ClassifierTrain(object):
    # 申明相关的属性
    def __init__(self, train_features, train_class):
        self.train_features = train_features
        self.train_class = train_class

    def SVM(self):
        ## Pipeline+GridSearchCV
        parameters = [
            {
                'pca__n_components':[10, 15, 20, 25, 30],
                'svm__kernel':['rbf'],
                'svm__gamma':[1e-3, 1e-2, 1e-1],
                'svm__C':[1e-2, 1e-1, 1, 5, 10]
            },
            {
                'pca__n_components':[10, 15, 20, 25, 30],
                'svm__kernel':['linear'],
                'svm__C':[1e-2, 1e-1, 1, 5, 10]
            }
        ]
        # parameters = {
        #     'pca__n_components':[10, 15, 20, 25, 30],
        #     'svm__kernel':['rbf'],
        #     'svm__gamma':[1e-3, 1e-2, 1e-1],
        #     'svm__C':[1e-2, 1e-1, 1, 5, 10]
        # }
        # print list(ParameterGrid(parameters))
        pipeline = Pipeline(
            steps = [
                ('pca', PCA()), # 'pca'对应'pca__'
                ('svm', SVC()) # 'svm'对应'svm__'
            ]
        )
        clf = GridSearchCV(
            estimator = pipeline,
            param_grid = parameters,
            cv = StratifiedKFold(self.train_class, 5),
            scoring = "accuracy",
            n_jobs = 3
        )
        clf.fit(self.train_features, self.train_class)
        best_clf = clf.best_estimator_
        return best_clf

    # def SVM(self):
    #     clf = SVC()
    #     clf.fit(self.train_features, self.train_class)
    #     best_clf = clf
    #     return best_clf

    def LibSVM(self):
        clf = LinearSVC()
        clf.fit(self.train_features, self.train_class)
        best_clf = clf
        return best_clf

def ClassifierTest(best_clf, test_features):
    test_class = best_clf.predict(test_features)
    return test_class


