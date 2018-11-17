import numpy as np
import tensorflow as tf

'''parameters
ui_mat = ui_matrix
avg = rating_mean
n_i = i_num
n_u = u_num
n_k = k
lr = 0.01
lamb_u = 0.02
lamb_v = 0.02
'''

#ra = SVD_plusplus_pred(ui_matrix,rating_mean,i_num,u_num,k)
#print(ra[2][13:30])

#data = ac.get_FilmTrust()

#ui_matrix,is_rating = ac.get_User_Item_matrix(data)

#ra = mf.MF_pred(ui_matrix,is_rating)
#print(ra[2][13:30])

#ilist = [13,14,15,16,1,18,19,20]
#ra_2_to_ilist = svd.SVD_pred_with_sim(ui_matrix,2,ilist)
#print(ra_2_to_ilist)

#import Algorithm.embedding as eb

#u,v = eb.SVD_ItemUser2vec(ui_matrix)

from Configurations.config import Config
from Tool.Dao import RationgDao
from Algorithm.rating.MF import BasicMatrixFactorization
from Tool.File import FileIO
from Algorithm.rating.SVD import SingularValueDecomposition
config = Config('./configurations/SVD.conf')
dao = RationgDao(config)

#mf = MatrixFactorization(config,dao)
#mf.showRecommenderInfo()

#mf.Training()
#mf.Testing()
svd = SingularValueDecomposition(config,dao)
svd.Training()
svd.Testing()

'''
print(dao.trainingSet.dataset[:10])
print(type(dao.trainingSet.dataset))
print(len(dao.trainingSet.dataset))
print(dao.trainingSet.item_num)
print(dao.trainingSet.user_num)
print(dao.trainingSet.item_means)
print(dao.trainingSet.user_means)
print(dao.trainingSet.items_RratedByUser['1'])
print(dao.trainingSet.users_RratedItem['1'])'''


#print(dao.trainingSet.id2user[3])

#print(dao.trainingSet.)
'''
print(dao.testingData[:10])
print(type(dao.testingData))
print(len(dao.testingData))

print(dao.validationData[:10])
print(type(dao.validationData))
print(len(dao.validationData))'''