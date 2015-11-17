# coding: utf-8
from __future__ import division
__author__ = 'LiNing'

import ConfigParser
import os
import re
import jieba
from MongoDBIO import MongoDBIO


def MakeWordsSet(words_file):
    fp = open(words_file, 'r')
    words_set = set()
    for line in fp.readlines():
        word = line.strip().decode("utf-8")
        if len(word)>0 and word not in words_set: # 去重
            words_set.add(word)
    fp.close()
    return words_set

country_list = [ u"中国", u"其他", u"美国", u"日本", u"英国", u"德国", u"澳大利亚", u"法国", u"韩国", u"意大利", u"西班牙"]
number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
Country_Number_Map = dict(zip(country_list, number_list))
Number_Country_Map = dict(zip(number_list, country_list))

Config_Dir = "./Config"
assert(os.path.exists(Config_Dir))
config_file = os.path.join(Config_Dir, "config.ini")
assert(os.path.exists(config_file))

Datas_Dir = os.path.join(Config_Dir, "datas")
all_sorted_words_list_file = os.path.join(Datas_Dir, "all_sorted_words_list")
train_datas_file = os.path.join(Datas_Dir, "train_datas")
Fea_Dict_Dir = os.path.join(Datas_Dir, "fea_dict")
Model_Dir = os.path.join(Datas_Dir, "model")
best_clf_file = os.path.join(Model_Dir, "best_clf")

Dict_Dir = os.path.join(Config_Dir, "dict")
dict_file = os.path.join(Dict_Dir, "dict")
if os.path.exists(dict_file):
    jieba.set_dictionary(dict_file) # 主词典
user_dict_file = os.path.join(Dict_Dir, "user_dict")
if os.path.exists(user_dict_file):
    jieba.load_userdict(user_dict_file) # 用户词典

Stopwords_Dir = os.path.join(Config_Dir, "stopwords")
assert(os.path.exists(Stopwords_Dir))
stopwords_file = os.path.join(Stopwords_Dir, "stopwords")
notstopwords_file = os.path.join(Stopwords_Dir, "notstopwords")
assert(os.path.exists(stopwords_file))
assert(os.path.exists(notstopwords_file))

## --------------------------------------------------------------------------------
conf = ConfigParser.ConfigParser()
conf.read(config_file)
host = conf.get("database", "host")
port = conf.getint("database", "port")
name = conf.get("database", "name")
password = conf.get("database", "password")
database = conf.get("database", "database")
collection = conf.get("database", "collection")

time_col = conf.get("database", "time_col")
content_col = conf.get("database", "content_col")
source_col = conf.get("database", "source_col")
t_status_col = conf.get("database", "t_status_col")
keyword_col = conf.get("database", "keyword_col")
country_col = conf.get("database", "country_col")
imp_col = conf.get("database", "imp_col")
limit_number = conf.getint("database", "limit_number")

lag = conf.get("dict", "lag")
fea_dict_size = conf.getint("dict", "fea_dict_size")

smtp_server = conf.get("email", "smtp_server")
from_addr = conf.get("email", "from_addr")
passwd = conf.get("email", "passwd")
to_addr = conf.get("email", "to_addr")
from_addr = conf.get("email", "from_addr")
sendmail_flag = conf.getboolean("email", "sendmail_flag")

test_speedup = conf.getboolean("other", "test_speedup")

print host, port, name, password, database, collection, \
    time_col, content_col, source_col, t_status_col, keyword_col, country_col, imp_col, \
    limit_number
print lag, fea_dict_size
to_addr = re.split(r',', re.search(r'\[(.*?)\]', to_addr).group(1).replace(' ', '').replace('"', ''))
print smtp_server, from_addr, passwd, to_addr, from_addr, sendmail_flag
print test_speedup

stopwords_set = MakeWordsSet(stopwords_file)
notstopwords_set = MakeWordsSet(notstopwords_file)

## --------------------------------------------------------------------------------
para = (host, port, name, password, database, collection)
posts = MongoDBIO(*para).Connection()
