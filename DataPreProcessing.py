# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from ReadBulletScreen import BulletScreen
from collections import OrderedDict



class DataPreProcessing(object):
    def __init__(self):
        self.docSet=[]

    '''
        *function name : addRestComment
        *函数功能:
        	-添加剩余的弹幕
        *输出:
    	    -docSet:  返回格式[[u'伪装', u'着看', u'完', u'僵尸', u'王', u'欢乐颂', u'取', u'汁源', u'诚信', u'发', u'全集', u'私', u'私', u'威信', u'来来来', u'欢乐颂', u'全集', u'超清', u'微信', u'欢乐颂',
                      将最后一个时间片的内容添加到docSet这种
        '''
    def addRestComment(self):
        doc=[]
        while (len(self.lines) != 0):
                for item in self.lines[0]["text"]:
                    doc.append(item)
                self.lines.pop(0)
        self.docSet.append(doc)



    '''
    *function name : sliceWithTime
    *输入参数:
    	-timeInterval : 每一个时间片的大小
    	-file_name : 要分析的视频的文件名
    	-time_length:分析视频的时间长度长度
    	-POS_tag:用于jieba分词的词性过滤
    *输出:
    	-docSet:  返回格式[[u'伪装', u'着看', u'完', u'僵尸', u'王', u'欢乐颂', u'取', u'汁源', u'诚信', u'发', u'全集', u'私', u'私', u'威信', u'来来来', u'欢乐颂', u'全集', u'超清', u'微信', u'欢乐颂',
                  每个时间片为一个数组,包含了时间片内若有的单词
    *函数功能:
    	-根据时间片的大小 将每一个时间片的视频弹幕划分为每一个数组
    '''
    def sliceWithTime(self,timeInterval,file_name,time_length,POS_tag):
        self.lines,vocabulary=BulletScreen().run(file_name,POS_tag)
        preTime=0
        lastTime=preTime+timeInterval
        for index in xrange(int(time_length/timeInterval)):
            doc =[]
            while(len(self.lines)!=0):
                if self.lines[0]["time"] <=lastTime:
                    for item in self.lines[0]["text"]:
                        doc.append(item)
                    self.lines.pop(0)
                else:
                    preTime=lastTime
                    lastTime=preTime+timeInterval
                    self.docSet.append(doc)
                    break

            print "doc size %d" % len(doc)
        print "doc size %d" % len(self.lines)
        self.addRestComment()
        #print len(self.docSet)
        #self.print_docSet(self.docSet)
        return self.docSet


#返回格式
#[[u'伪装', u'着看', u'完', u'僵尸', u'王', u'欢乐颂', u'取', u'汁源', u'诚信', u'发', u'全集', u'私', u'私', u'威信', u'来来来', u'欢乐颂', u'全集', u'超清', u'微信', u'欢乐颂',
if __name__=="__main__":
    #时间片大小、单位秒
    timeInterval = 100
    # 所要分析的弹幕文件
    file_name = "data/1.txt"
    # 所要分析弹幕文件的时间长度
    time_length = 2582
    # 采用词性过滤的方式来过滤对弹幕挖掘没有实际意义的词 具体可查 http://www.cnblogs.com/adienhsuan/p/5674033.html
    POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul",
               "r", "eng"]
    print DataPreProcessing().sliceWithTime(timeInterval,file_name,time_length,POS_tag)



