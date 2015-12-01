#coding: utf-8
from __future__ import division
__author__ = 'LiNing'

import pymongo


class MongoDBIO(object):
    # 申明相关的属性
    def __init__(self, host, port, name, password, database, collection):
        self.host = host
        self.port = port
        self.name = name
        self.password = password
        self.database = database
        self.collection = collection

    # 连接数据库，db和posts为数据库和集合的游标
    def Connection(self):
        ## --------------------------------------------------------------------------------
        # # connection = pymongo.Connection() # 连接本地数据库
        # connection = pymongo.Connection(host=self.host, port=self.port)
        # db = connection[self.database]
        # if len(self.name)>0:
        #     db.authenticate(name=self.name, password=self.password) # 验证用户名密码
        # else:
        #     pass
        ## --------------------------------------------------------------------------------
        # mongodb://[username:password@]host1[:port1][,host2[:port2],...[,hostN[:portN]]][/[database][?options]]
        if len(self.name)>0:
            uri = "mongodb://%s:%s@%s:%d/%s" % (self.name, self.password, self.host, self.port, self.database)
        else:
            uri = "mongodb://%s:%d/%s" % (self.host, self.port, self.database)
        # print uri
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        ## --------------------------------------------------------------------------------
        print "Database:", db.name
        # print db.collection_names() # 查询所有集合
        posts = db[self.collection]
        print "Collection:", posts.name
        print "Count:", posts.count()

        return posts
