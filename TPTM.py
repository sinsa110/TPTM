import numpy as np
from DataPreProcessing import DataPreProcessing
import Eq
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
        self.mpre_c=[np.zeros(len(shot),self.K) for shot in self.shot_comments_vector]

        _eq=Eq()
        _eq._Eq2_term(self.self.shot_comments_vector)
        _eq._Eq3_term(self.self.shot_comments_vector)

        self._lambda = np.random.randn(self.v, self.K)
        self._x_c = np.random.randn(self.U, self.K)



    def _eta(self,i):
        return  0.1/(np.pow(2,i%10))


    def _calc_pi_c(self):



    def _Eq7(self,i):
        pass

    def _Eq8(self,i):
        pass

    def _Eq5(self):
        pass

    def _Eq6(self):
        pass

    def _Eq3(self):



    def _calc_(self):
        for i in range(1,self.iters+1):
            if i%200==0 and (i%2==0):
                self._Eq7()
            elif i%200==0 and (i%2!=0):
                self._Eq8()
            if i%200==0:
                self.update_m_pres_and_m_prec()











if __name__ == '__main__':
    TPTMModel(5)