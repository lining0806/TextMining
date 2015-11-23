# 文本挖掘系统 Text Mining System

#### Author: LiNing 
#### Email: lining0806@gmail.com
#### Blog: [宁哥的小站](http://www.lining0806.com/)

***


## 系统说明

* 集成了**文本关键词提取**的功能
* 集成了**文本过滤**及**邮件实时通知**的功能
* 集成了**文本分类**即**打标签**的功能
* 集成了**文本推荐**即**热点评价**的功能
* **支持中英文**


## 关于分词
**英文分词，采用nltk工具包进行分词**  

	pip install nltk 

**中文分词，采用jieba工具包进行分词**  

	pip install jieba 

**jieba分词**

	dict 主词典文件 
	user_dict 用户词典文件，即分词白名单 

**user_dict为分词白名单**
* 如果添加的过滤词（包括黑名单和白名单）无法正确被jieba正确分词，则将该需要添加的单词及词频加入到主词典dict文件中或者用户词典user_dict，一行一个（词频也可省略）  

## 关于过滤词与关键词

**停用词与非停用词**

	stopwords 停用词，即过滤词黑名单
	notstopwords 非停用词，即关键词白名单

**stopwords为过滤词黑名单**    
* 可以随时添加过滤的单词，一行一个  

**notstopwords为关键词白名单**  
* 可以随时添加关键的单词，一行一个  


## 关于配置

**config文件：**  
* 可以进行服务器配置，针对数据库中制订collection的不同字段column。 
* 可以限定操作数据库条目的数量，默认时间从最近往前推。
* 可以选择语言(中文，英文)。
* 可以设置特征词词典的维度，特征词的选取可以通过该词的词频数或者包含该词的文档频率来确定。特征词是专门用于计算文本特征的。
* 可以设置是否接收邮件通知。
* 可以设置版本加速，如果加速，此时会将文本特征词和模型固定化！

## 其他说明
* 更改分词文件dict user_dict 或者 lag
需要事先手动删除datas文件夹

* 更改训练集
需要事先手动删除all_words_dict和train_datas

* 更改过滤词文件stopwords notstopwords
重新运行程序即可

## 关于环境搭建

**Ubuntu下pil numpy scipy matplotlib的安装：**  

    sudo apt-get update
    sudo apt-get install git g++
    sudo apt-get install python-dev python-setuptools
    
    sudo easy_install Cython 
    sudo easy_install pil
    sudo apt-get install libatlas-base-dev # 科学计算库
    sudo apt-get install gfortran # fortran编译器
    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev
    export BLAS=/usr/lib/libblas/libblas.so 
    export LAPACK=/usr/lib/lapack/liblapack.so 
    export ATLAS=/usr/lib/atlas-base/libatlas.so
    sudo apt-get install python-numpy
    sudo apt-get install python-scipy
    sudo apt-get install python-matplotlib
    
    sudo easy_install jieba
    sudo easy_install scikit-learn
    sudo easy_install simplejson
    sudo easy_install pymongo


**CentOS下pil numpy scipy matplotlib的安装：**  

    离线安装Python
    sudo yum install python-setuptools
    sudo yum install gcc-gfortran 
    sudo yum install blas-devel
    sudo yum install lapack-devel
    进入numpy解压目录
    sudo python setup.py build
    sudo python setup.py install
    进入scipy解压目录
    sudo python setup.py build
    sudo python setup.py install
    进入matplotlib目录
    sudo yum install libpng-devel
    sudo python setup.py build
    sudo python setup.py install
    
    sudo easy_install jieba
    sudo easy_install scikit-learn
    sudo easy_install simplejson
    sudo easy_install pymongo
