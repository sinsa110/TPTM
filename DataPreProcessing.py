# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from ReadBulletScreen import BulletScreen
from collections import OrderedDict
import copy


class DataPreProcessing(object):
    def __init__(self):
        self.docSet=[]

    def addRestComment(self):
        doc=[]

        while (len(self.lines)!= 0):
                doc.append(self.lines[0])
                self.lines.pop(0)
        self.docSet.append(doc)


    def sliceWithTime(self,timeInterval,file_name,time_length,POS_tag):
        #self.lines,self.vocabulary=BulletScreen().run(file_name,POS_tag)
        self.lines = BulletScreen().run(file_name, POS_tag)
        preTime=0
        lastTime=preTime+timeInterval

        for index in xrange(int(time_length/timeInterval)):
            doc =[]
            while(len(self.lines)!=0):
                if self.lines[0]["time"] <=lastTime:
                    doc.append(self.lines[0])
                    self.lines.pop(0)
                else:
                    preTime=lastTime
                    lastTime=preTime+timeInterval
                    self.docSet.append(doc)
                    print "doc size %d" % len(doc)
                    doc = []
                    break
            if(len(self.lines)==0):
                self.docSet.append(doc)
        self.addRestComment()
        print self.docSet


    def init_vocabulary(self):
        self.vocabulary=OrderedDict()
        for shot in self.docSet:
            for comment in shot:
                for item in comment["text"]:
                    if item not in self.vocabulary:
                        self.vocabulary[item]=0






    #address user's comment and reformat self.docSet()
    def user_all_commnt(self,timeInterval,file_name,time_length,POS_tag):
        self.sliceWithTime(timeInterval, file_name, time_length, POS_tag)
        self.init_vocabulary()
        user={}
        shot_comments=[]
        _raw_comment=[]
        _shot_comments_vector=[]
        _comments_vector=[]
        _comment_2_user_matrix=[]
        for i,item in enumerate(self.docSet):
            _comment_2_user = []
            _comments_vector = []
            for j,comment in enumerate(item):
                if comment["user"] not in user:
                    user[comment["user"]]=[]
                    user[comment["user"]].append((i,j))
                else:
                    user[comment["user"]].append((i, j))
                _raw_comment.append([" ".join(comment["text"])])
                _vocabulary = self.vocabulary.copy()
                for item2 in comment["text"]:
                    _vocabulary[item2]+=1
                _comments_vector.append(_vocabulary.values())
                _comment_2_user.append(comment["user"])

            shot_comments.append(_raw_comment)
            _raw_comment=[]
            _shot_comments_vector.append(_comments_vector)
            _comment_2_user_matrix.append(_comment_2_user)

        return user,shot_comments,_shot_comments_vector,_comment_2_user_matrix



if __name__=="__main__":
    #时间片大小、单位秒
    timeInterval = 5
    # 所要分析的弹幕文件
    #file_name = "data/18942125.xml"
    file_name = "data/1.xml"
    # 所要分析弹幕文件的时间长度
    time_length =11
    # 采用词性过滤的方式来过滤对弹幕挖掘没有实际意义的词 具体可查 http://www.cnblogs.com/adienhsuan/p/5674033.html
    POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul","r", "eng"]
    t=DataPreProcessing()
    user_comment,shot_comments,shot_comments_vector,_comment_2_user_matrix=t.user_all_commnt(timeInterval,file_name,time_length,POS_tag)
    print shot_comments
    print shot_comments_vector

    #docSet
    # [[{'text': [u'娶', u'我爱你', u'你们好'], 'user': 'ef4a4195', 'lineno': 3, 'time': 5},
    #   {'text': [u'收看', u'字幕', u'卵'], 'user': 'f9498f82', 'lineno': 6, 'time': 5}],
    #  [{'text': [u'红牛', u'帽', u'明白', u'散', u'命', u'交出去', u'学'], 'user': 'f9498f82', 'lineno': 5, 'time': 8},
    #   {'text': [u'收看', u'字幕', u'吊'], 'user': '3ed492b0', 'lineno': 2, 'time': 9},
    #   {'text': [u'字幕', u'卵'], 'user': '728e21d2', 'lineno': 4, 'time': 9}], []]

    #user_comment  corresponig to the _x_u
    #{'728e21d2': [(1, 2)], 'f9498f82': [(0, 1), (1, 0)], 'ef4a4195': [(0, 0)], '3ed492b0': [(1, 1)]}

    #shot_comments
    # [[[u'娶 我爱你 你们好'], [u'收看 字幕 卵']], [[u'红牛 帽 明白 散 命 交出去 学'], [u'收看 字幕 吊'], [u'字幕 卵']], []]

    #shot_comments_vector
    #OrderedDict
    #[[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], []]

    #_comment_2_user_matrix
    #[['ef4a4195', 'f9498f81'], ['f9498f82', '3ed492b0', '728e21d2'], []]





