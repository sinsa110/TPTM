import numpy as np
from DataPreProcessing import DataPreProcessing
import Eq
from  LDA import LDAModel



import numpy as np
import random
import codecs
import os

from collections import OrderedDict



#文件路径
trainfile = "data/train.dat"
thetafile = "data/tmp/thetafile.txt"
phifile = "data/tmp/phifile.txt"
wordidmapfile = "data/tmp/wordidmap.txt"
topNfile = "data/tmp/topNfile.txt"
tassginfile = "data/tmp/tassginfile.txt"
#模型初始参数
K = 10
alpha = 0.1
beta = 0.1
iter_times = 10
top_words_num = 3


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class DataPreProcessing(object):

    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w','utf-8') as f:
            for word,id in self.word2id.items():
                f.write(word +"\t"+str(id)+"\n")


class TPTMModel(object):

    def __init__(self,K=5,iters=1000):


        #the numer of topics in a video
        self.K=K
        self.user_comment, self.shot_comments,self.shot_comments_vector,self._comment_2_user_matrix\
            = DataPreProcessing().user_all_commnt(timeInterval, file_name, time_length, POS_tag)

        #the number of shots in a video
        self.v=len(shot_comments)

        #the number of users in a video
        self.U=len(user_comment)

        #the number of iterations
        self.iters=iters

        self.vocabulary_length=len(self.shot_comments_vector)
        self.gamma_s=1

        self.gamma_c=1

       # v*c*K initialize mpre_c to 0
        self._mpre_c=np.array([np.zeros(len(shot),self.K) for shot in self.shot_comments_vector])

        _eq=Eq()
        _eq._Eq2_term(self.self.shot_comments_vector)
        _eq._Eq3_term(self.self.shot_comments_vector)

        self._lambda = np.random.randn(self.v, self.K)
        self._x_c = np.random.randn(self.U, self.K)

        self.save_data_file()




        #init lda
        self.dpre = dpre  # 获取预处理参数
        #
        # 模型参数
        # 聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        #
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num
        #
        # 文件变量
        # 分好词的文件trainfile
        # 词对应id文件wordidmapfile
        # 文章-主题分布文件thetafile
        # 词-主题分布文件phifile
        # 每个主题topN词文件topNfile
        # 最后分派结果文件tassginfile
        # 模型训练选择的参数文件paramfile
        #
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile

        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # ndsum,每各doc中词的总数
        self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")
        self.nwsum = np.zeros(self.K, dtype="int")
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")
        self.Z = np.array(
            [[0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])  # M*doc.size()，文档中词的主题分布

        # 随机先分配类型
        for x in xrange(len(self.Z)):
            self.ndsum[x] = self.dpre.docs[x].length
            for y in xrange(self.dpre.docs[x].length):
                topic = random.randint(0, self.K - 1)
                self.Z[x][y] = topic
                self.nw[self.dpre.docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1

        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])
        self.phi = np.array([[0.0 for y in xrange(self.dpre.words_count)] for x in xrange(self.K)])





    def save_data_file(self,file_name="data/train.dat"):
        with open(file_name,"w") as f:
            for shot in self.shot_comments:
                for comments in shot:
                    f.write(" ".join(comments))
                    f.write("\n")



    def _eta(self,i):
        return  0.1/(np.pow(2,i%10))

    # _comment_2_user_matrix
    # [['ef4a4195', 'f9498f81'], ['f9498f82', '3ed492b0', '728e21d2'], []]
    def _calc_pi_c(self):
        _phi=[]
        for i,users in enumerate(self._comment_2_user_matrix):
            _phi_2=[]
            for j,user in enumerate(users):
                x_u=_x_u[self.user_comment.keys().index(user)]
                _phi_2.append(self._lambda[i]*x_u+self._mpre_c[i][j])
            _phi.append(_phi_2)
        return np.array(_phi)



    # def _Eq7(self,i):
    #     pass
    #
    # def _Eq8(self,i):
    #     pass
    #
    # def _Eq5(self):
    #     pass
    #
    # def _Eq6(self):
    #     pass
    #
    # def _Eq3(self):



    def _calc_(self):
        for i in range(1,self.iters+1):
            if i%200==0 and (i%2==0):
                self._calc_pi_c()
                Eq._Eq7()
            elif i%200==0 and (i%2!=0):
                Eq._Eq8()
            if i%200==0:
                Eq._Eq2()
                Eq._Eq3()












if __name__ == '__main__':
    TPTMModel(5)