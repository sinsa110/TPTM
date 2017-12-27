# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from ReadBulletScreen import BulletScreen
from collections import OrderedDict
import copy

try:
    import cPickle as pickle
except ImportError:
    import pickle



class BulletPreProcessing(object):
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
        print len(self.lines)
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
                    #print "doc size %d" % len(doc)
                    doc = []
                    break
            if(len(self.lines)==0):
                self.docSet.append(doc)

        if(len(self.lines)!=0):
            self.addRestComment()
        print self.docSet


    def init_vocabulary(self):
        self.vocabulary=OrderedDict()
        for shot in self.docSet:
            for comment in shot:
                for item in comment["text"]:
                    if item not in self.vocabulary:
                        self.vocabulary[item]=0


    def count_comment_number_shot(self):
        shot_number=[]
        for doc in self.docSet:
            shot_number.append(len(doc))
        return shot_number






    #address user's comment and reformat self.docSet()
    def user_all_comment(self,timeInterval,file_name,time_length,POS_tag):
        self.sliceWithTime(timeInterval, file_name, time_length, POS_tag)
        self.init_vocabulary()
        print "zeze"
        shot_number=self.count_comment_number_shot()
        print shot_number

        user={}
        shot_comments=[]
        _raw_comment=[]
        _shot_comments_vector=[]
        _comments_vector=[]
        _comment_2_user_matrix=[]
        print len(self.docSet)
        for i,item in enumerate(self.docSet):
            _comment_2_user = []
            _comments_vector = []
            print "i index"+str(i)
            for j,comment in enumerate(item):
                print "j index"+str(j)
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

        return user,shot_comments,_shot_comments_vector,_comment_2_user_matrix,shot_number,self.vocabulary




def save_data_file(shot_comments,file_name="data/train.txt"):
        with open(file_name,"w") as f:
            for shot in shot_comments:
                for comments in shot:
                    f.write(" ".join(comments))
                    f.write("\n")


def store(user_comment,shot_comments,shot_comments_vector,_comment_2_user_matrix,shot_comemnt_number,vocabulary):
    fw = open("data/cache/user_comment", "wb")
    pickle.dump(user_comment, fw)
    fw.close()
    fw = open("data/cache/shot_comments", "wb")
    pickle.dump(shot_comments, fw)
    fw.close()
    fw = open("data/cache/shot_comments_vector", "wb")
    pickle.dump(shot_comments_vector, fw)
    fw.close()
    fw = open("data/cache/_comment_2_user_matrix", "wb")
    pickle.dump(_comment_2_user_matrix, fw)
    fw.close()
    fw = open("data/cache/shot_comemnt_number", "wb")
    pickle.dump(shot_comemnt_number, fw)
    fw.close()
    fw = open("data/cache/vocabulary", "wb")
    pickle.dump(vocabulary, fw)
    fw.close()



if __name__=="__main__":
    #时间片大小、单位秒
    timeInterval = 300
    # 所要分析的弹幕文件
    file_name = "data/huanlesong.xml"
    # file_name = "data/1.xml"
    # time_length = 11
    time_length=2582

    # 采用词性过滤的方式来过滤对弹幕挖掘没有实际意义的词 具体可查 http://www.cnblogs.com/adienhsuan/p/5674033.html
    POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul","r", "eng"]
    t=BulletPreProcessing()
    user_comment,shot_comments,shot_comments_vector,_comment_2_user_matrix,shot_comemnt_number,vocabulary=\
        t.user_all_comment(timeInterval,file_name,time_length,POS_tag)
    gi(user_comment,shot_comments,shot_comments_vector,_comment_2_user_matrix,shot_comemnt_number,vocabulary)
    save_data_file(shot_comments)
    # print shot_comments
    # print shot_comments_vector
    # print shot_comemnt_number
    # for i in range(1, len(shot_comemnt_number)):
    #     shot_comemnt_number[i] += shot_comemnt_number[i - 1]
    print shot_comemnt_number

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

    #shot_comemnt_number
    #[2, 3, 0]





