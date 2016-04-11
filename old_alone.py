# -*- coding: utf-8 -*-
'''
Created on 2016. 2. 24.

@author: DS
'''
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
from sklearn import svm
from numpy import newaxis
from pip.req.req_file import preprocess
from mistune import preprocessing
from sklearn.linear_model.base import LinearRegression
sns.set()


#첫번째 행을 columns index로 파일읽기
data = pd.read_csv('raw_data_number.csv', header=0, encoding='cp949')
print(data.tail())

# 눈으로 자료의 관계성 확인
# 질문? plt 조절
# sns.pairplot(data, diag_kind="kde", kind="reg" ,vars = ['older', 'disable', 'bank', 'seniorcenter', 'movingin', 'movingout'], size=1.5)
# plt.show()
#http://bokeh.pydata.org/en/latest/search.html?q=pairplot
# 강의자료 visualization 참조

# 1.Formulation (x,y 정해주기)################################################
y = data['older']
x = data.drop(labels = ['code', 'gu', 'dong', 'older'], axis=1)
# print("seniorcenter 복지시설 갯수 / bank 은행 갯수 / fire 화재사건 건수 / older 독거노인 수 / minratio 전입인구 수 비율 / moutratio 전출인구 수 비율 / gasratio 도시가스 이용비율 / dratio 장애인 인구 비율")

# correlation 확인
# print(data.corr())
# plt.imshow(data.corr(), interpolation="none")
# plt.grid(False)
# plt.show()

# 2. scaling, normalize ################################################
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
x_scale = pd.DataFrame(scale(x), index = x.index, columns = x.columns)
x_scale_norm = pd.DataFrame(normalize(x_scale), columns = x.columns)

#scaling => normalize 된 것의 pairplot을 위해 잠시 y를 붙인 x_test를 만듦
x_test = pd.concat([x_scale_norm, y], axis=1)
print(x_test.tail())
# sns.pairplot(x_test, diag_kind="kde", kind="reg" ,vars = ['older', 'disable', 'bank', 'seniorcenter', 'movingin', 'movingout'], size=1.5)
# plt.show()

# 3. modeling, estimation, prediction (OLS regression) ##########################################
import statsmodels.api as sm

# 3-1 결과값이 향상된다면 intercept 없어 된다!!!
model_OLS = sm.OLS(y,x_scale_norm) 
result_OLS = model_OLS.fit()
print(result_OLS.summary())

# 3-2 intercept 추가
x_intercept = sm.add_constant(x_scale_norm)
model_OLS = sm.OLS(y,x_intercept) 
result_OLS = model_OLS.fit()
print(result_OLS.summary())
# Outlier를 feature selecting 전에 할 수도 있다.

# 3-3 feature selecting
print("##After feature selecting##\n")
x_intercept = x_intercept.drop(labels = ['bank', 'movingout', 'seniorcenter', 'movingin'], axis=1)
model_OLS = sm.OLS(y,x_intercept) 
result_OLS = model_OLS.fit()
print(result_OLS.summary())

# 4. Outlier ############################################################
# 눈으로 Outlier 파악해보기
# fig, ax = plt.subplots(figsize=(15, 10))
# sm.graphics.influence_plot(result_OLS, plot_alpha=0.3, ax=ax)
# plt.show()

idx_outlier = np.nonzero(result_OLS.outlier_test().ix[:, -1].abs() < 0.01)[0]
print("Outlier : ", idx_outlier)

x_intercept_out = x_intercept.drop(idx_outlier)
y_out = y.drop(idx_outlier)
model_OLS_out = sm.OLS(y_out, x_intercept_out)
result_OLS_out = model_OLS_out.fit()
print(result_OLS_out.summary())

# R-square 눈으로 확인하기
# plt.scatter(x_intercept_out.ix[:, -1], result_OLS_out.fittedvalues)
# plt.show()

# 5. Closs-validation(K-Fold) ############################################
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

x_out = x_scale_norm.drop(idx_outlier)
print("r2_score : " , cross_val_score(LinearRegression(), x, y, "r2", KFold(len(x_scale_norm))))
print("mean_squared_error : " , cross_val_score(LinearRegression(), x, y, "mean_squared_error", KFold(len(x_scale_norm))))

# SVR(Support Vector Regression), Radial Basis Function (RBF) kernel SVM #########################
print("\n ##Support Vector Regression Score##")
x = x.drop(labels = ['bank', 'movingout', 'seniorcenter', 'movingin'], axis=1)

from sklearn.svm import SVR
svr_linear = SVR(kernel='linear')

x = x.drop(idx_outlier)
y = y.drop(idx_outlier)
print("SVR_linear model(score)  : ", svr_linear.fit(x,y).score(x,y))

#갑자기 이곳에서 무한하게 작업된다???????????????????????????? 
svr_poly = SVR(kernel='poly')
print("SVR_poly model(score)    : ", svr_poly.fit(x,y).score(x,y))
#   
# svr_sigmoid = SVR(kernel='sigmoid')
# print("SVR_sigmoid model(score) : ", svr_sigmoid.fit(x,y).score(x,y))
#   
# svr_rbf = SVR(kernel='rbf')
# print("SVR_rbf model(score)     : ", svr_rbf.fit(x,y).score(x,y))

#눈으로 보고 싶은데....
# x = np.array(x['disable'])
# xx = np.linspace(-1, 5, 100)[:, np.newaxis]
# yy = svr_linear.fit(x, y).predict(xx)
# plt.figure(figsize=(10, 8))
# plt.scatter(x, y, c='k', label='data')
# plt.hold('on')
# plt.plot(xx, yy, c='r', label='linear')
# plt.show()

# from scipy.optimize import curve_fit
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.optimize.curve_fit.html
# plt.show()