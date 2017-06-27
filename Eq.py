import numpy as np
from scipy.special import digamma

try:
    import cPickle as pickle
except ImportError:
    import pickle

def grab_Eq2_term():
    fr = open("data/cache/_Eq2_term", "rb")
    term = pickle.load(fr)
    fr.close()
    return term

def grab_Eq3_term():
    fr = open("data/cache/_Eq3_term", "rb")
    term = pickle.load(fr)
    fr.close()
    return term

def store_Eq2_term(term):
    fw = open("data/cache/_Eq2_term", "wb")
    pickle.dump(term, fw)
    fw.close()


def store_Eq3_term(term):
    fw = open("data/cache/_Eq3_term", "wb")
    pickle.dump(term, fw)
    fw.close()


# def cos(vector1,vector2):
#         return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def lgt(y):
    return np.log(1+np.exp(y))

def dlgt(y):
    return np.exp(y)/(1+np.exp(y))


h=np.array([[[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]],[[1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 1],[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]])
class Eq(object):

    #[[[1.0], [0.36787944117144233]], [[1.0], [0.36787944117144233], [0.1353352832366127, 0.36787944117144233], [0.049787068367863944, 0.1353352832366127, 0.36787944117144233]], [[1.0], [0.36787944117144233]]]
    def _Eq2_term(self,shot_comments_vector):
        _term=[]
        for index,item in enumerate(shot_comments_vector):
            if index==0:
                _term.append([1.0])
            else:
                temp=[]
                for index2,term_pre in enumerate(shot_comments_vector[:index]):
                    temp.append(np.exp(-1.0*(np.abs(index-index2))))
                _term.append(temp)
        store_Eq2_term(_term)
        return _term

    #V*K
    # [[0.          0.          0.          0.]
    #  [3.18263406  4.17975763  2.45225561  3.99634046]
    #  [1.70958045  2.76126519  2.14395926  2.12047888]]
    def _Eq2(self,_lambda,K):
        _term=grab_Eq2_term()
        _term_sum=[np.sum(term) for term in np.array(_term)]
        _m_pre_s=[]
        for index,item in enumerate(_term):
            if index==0:
                _m_pre_s.append(np.array([0.0]*K))
            else:
                total=np.ones(K)
                for index2,term2  in enumerate(item):
                    total+=_lambda[index2]*term2
                _m_pre_s.append(total/_term_sum[index])
        return np.array(_m_pre_s)

    #_comment_2_user_matrix
    #[['ef4a4195', 'f9498f81'], ['f9498f82', '3ed492b0', '728e21d2'], []]


    # user_comment
    # {'728e21d2': [(1, 2)], 'f9498f82': [(0, 1), (1, 0)], 'ef4a4195': [(0, 0)], '3ed492b0': [(1, 1)]}
    def _Eq5(self,_lambda,K,_x_u,m_pre_c,shot_comments_vector,user_comment,_comment_2_user_matrix):
        m_pre_s=self._Eq2(_lambda,K)
        #V*C
        #cacluate _term_2   comment in every shot
        total=[]
        for i,users in enumerate(_comment_2_user_matrix):

            #sum all comment in one shot
            _rows=np.zeros(K)
            for j,user in enumerate(users):
                #x_u
                x_u=_x_u[user_comment.keys().index(user)]
                _rows=x_u*dlgt(x_u*_lambda[i]+m_pre_c[i][j])* \
                (digamma(np.sum(lgt(x_u*_lambda[i]+m_pre_c[i][j])))\
                 -digamma(np.sum(lgt(x_u*_lambda[i]+m_pre_c[i][j]))+np.sum(shot_comments_vector[i][j]))\
                 +digamma(lgt(x_u*_lambda[i]+m_pre_c[i][j]))\
                 -digamma(lgt(x_u*_lambda[i]+m_pre_c[i][j])))

            total.append(_rows)

        _term = -1 * _lambda - m_pre_s+np.array(total)
        return _term



    def _Eq3(self,_pi,K):
        pass



    def _Eq3_term(self,shot_comments_vector):
        _shot_comment=[]
        for shot in shot_comments_vector:
            _shot=[]
            for index,comment in enumerate(shot):
                if index==0:
                    _shot.append([1.0])
                else:
                    temp=[]
                    for index2,term_pre in enumerate(shot[:index]):
                        temp.append(np.exp(-1*np.abs(index2-index)))
                    _shot.append(temp)
            _shot_comment.append(_shot)
        store_Eq3_term(_shot_comment)
        return _shot_comment




    def _Eq5(self,_lambda):
        pass




if __name__ == '__main__':

    #print Eq()._Eq2_term(h)
    #print Eq()._Eq3_term(h)
    K=4
    _lambda=np.random.randn(3,K)
    print Eq()._Eq2(_lambda,K)











