import numpy as np
from DataPreProcessing import DataPreProcessing

class TPTMModel(object):
    def __init__(self,K=5,iters=1000):


        #the numer of topics in a video
        self.K=K
        self.user_comment, self.shot_comments,self.shot_comments_vector = DataPreProcessing().\
            user_all_commnt(timeInterval, file_name, time_length, POS_tag)

        #the number of shots in a video
        self.v=len(shot_comments)

        #the number of users in a video
        self.U=len(user_comment)

        #the number of iterations
        self.iters=iters

        #
        self.gamma_s=1

        self.gamma_c=1


    def initialize_lambda_and_x(self):

        self._lambda=np.random.randn(self.v,self.K)
        self._x=np.random.randn(self.U,self.K)


    def _eta(self,i):
        return  0.1/(np.pow(2,i%10))


    def _Eq7(self,i):
        pass

    def _Eq8(self,i):
        pass

    def _Eq5(self):
        pass

    def _Eq6(self):
        pass


    def _Eq2(self):
        pass

    def _Eq3(self):
        pass

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