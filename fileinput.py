# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
from pandas.io.pytables import IndexCol
from pandas.core.frame import DataFrame
from requests.api import head
from sklearn.linear_model.base import LinearRegression
from sklearn import svm
from numpy import newaxis
from pip.req.req_file import preprocess
from mistune import preprocessing
sns.set()

#첫번째 행을 columns index로 파일읽기
#seniorcenter 복지시설 갯수 bank 은행 갯수 fire 화재사건 건수 older 독거노인 수 ,minratio 전입인구 수 비율(동), moutratio 전출인구 수 비율(동) ,gasratio 도시가스 이용비율(동, 세대별), dratio 장애인 인구 비율(동)
raw_data = pd.read_csv('raw_data.csv', header=0, encoding='cp949')
# print(raw_data.tail())

#ix Column을 label slicing 로 필요한 것만 슬라이싱
data = raw_data.ix[:,'movingin':].drop(labels = ['movingin', 'movingout', 'gas', 'disable', 'totalpopulation(2010)', 'orderratio'], axis=1)
print(data.tail())

y = data['older']
x = data.drop(labels = ['older'], axis=1)
# print("seniorcenter 복지시설 갯수 / bank 은행 갯수 / fire 화재사건 건수 / older 독거노인 수 / minratio 전입인구 수 비율 / moutratio 전출인구 수 비율 / gasratio 도시가스 이용비율 / dratio 장애인 인구 비율")

# 눈으로 자료의 관계성 확인
# 질문? plt 조절
# sns.pairplot(data, diag_kind="kde", kind="reg" ,vars = ['older', 'seniorcenter', 'bank', 'movingin ration', 'movingout ration', 'gasratio'], size=1.5)
# plt.show()
#http://bokeh.pydata.org/en/latest/search.html?q=pairplot
# 강의자료 visualization 참조

# correlation 확인
# print(data.corr())

#scaling, normalize, array 변환 막기!!!
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
x_scale = pd.DataFrame(scale(x), index = x.index, columns = x.columns)
x_scale_norm = pd.DataFrame(normalize(x_scale), index = x.index, columns = x.columns)

#scaling => normalize 된 것의 pairplot을 위해 잠시 y를 붙인 x_test를 만듦
x_test = pd.concat([x_scale_norm, data['older']], axis=1)
sns.pairplot(x_test, diag_kind="kde", kind="reg" ,vars = ['older', 'seniorcenter', 'bank', 'movingin ration', 'movingout ration', 'gasratio'], size=1.5)
plt.show()

#질문 array를 dataframe 에 넣는법???
# clm_list = []
# for colnumn in data_scale.columns:
#     clm_list.append(colnumn)
# print(clm_list)
#질문 dataframe iterator
# print("\n data_scale :  %0.2f" % data_scale)

# correlation 확인
# print(x_scale.corr())

#OLS 회귀분석!#######################################################################
import statsmodels.api as sm

# 결과값이 향상된다면 intercept 없어 된다!!!
model_OLS = sm.OLS(y,x_scale_norm) 
result_OLS = model_OLS.fit()
print(result_OLS.summary2())

#intercept 추가
x_intercept = sm.add_constant(x_scale_norm)
model_OLS = sm.OLS(y,x_intercept) 
result_OLS = model_OLS.fit()
print(result_OLS.summary2())

# corealtion이 작은 피처제거
x_intercept = x_intercept.drop(labels = ['bank', 'movingin ration', 'gasratio'], axis=1)
model_OLS = sm.OLS(y,x_intercept) 
result_OLS = model_OLS.fit()
print(result_OLS.summary2())

# SVR(Support Vector Regression), Radial Basis Function (RBF) kernel SVM #########################
print("##Support Vector Regression Score##")
from sklearn.svm import SVR
svr_sigmoid = SVR(kernel='linear')
print("SVR_linear model(score)  : ", svr_sigmoid.fit(x,y).score(x,y))

svr_sigmoid = SVR(kernel='poly')
print("SVR_poly model(score)    : ", svr_sigmoid.fit(x,y).score(x,y))

svr_sigmoid = SVR(kernel='sigmoid')
print("SVR_sigmoid model(score) : ", svr_sigmoid.fit(x,y).score(x,y))

svr_sigmoid = SVR(kernel='rbf')
print("SVR_rbf model(score)     : ", svr_sigmoid.fit(x,y).score(x,y))


# from scipy.optimize import curve_fit
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.optimize.curve_fit.html
# plt.show()
################## 

#피처하나들의 관계를 확인해봅니다
#array를 2차원 vector로 차원확장
# x_seniorcenter = x['seniorcenter'][:, np.newaxis]
# x_fire = x['fire'][:, np.newaxis]
# print("senior center(score) : ", svr_sigmoid.fit(x_seniorcenter,y).score(x_seniorcenter,y))
# print("fire(score) : ", svr_sigmoid.fit(x_fire,y).score(x_fire,y))

#cross validation
# from sklearn.cross_validation import train_test_split, KFold
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# x_train_intercept = sm.add_constant(x_train)
# model_old_OLS2 = sm.OLS(y_train, x_train_intercept) 
# result_OLS2 = model_old_OLS2.fit()
# print(result_OLS2.summary2())

