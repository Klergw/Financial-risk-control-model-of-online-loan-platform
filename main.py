import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from EDA_and_data_cleaning import EDA_and_data_cleaning
from preprocessing_and_feature_engineering import preprocessing_and_feature_engineering
from feature_selection import feature_selection
from single_model_xgb import single_model_xgb
from single_model_lgb import single_model_lgb
from model_ensemble import model_ensemble
path='./data/'
graph=True
#合并训练集和测试集，统一进行数据处理
train_master0 = pd.read_csv(path+'Master_Training_Set.csv',encoding='gbk')
train_userupdateinfo0 = pd.read_csv(path+'Userupdate_Info_Training_Set.csv',encoding='gbk')
train_loginfo0 = pd.read_csv(path+'LogInfo_Training_Set.csv',encoding='gbk')

train_master1 = pd.read_csv(path+'Master_Test_Set.csv',encoding='gbk')
#给测试集加上target数据，便于后续统一处理
train_master1['target']=train_master0['target']
train_userupdateinfo1 = pd.read_csv(path+'Userupdate_Info_Test_Set.csv',encoding='gbk')
train_loginfo1 = pd.read_csv(path+'LogInfo_Test_Set.csv',encoding='gbk')
train_master=pd.concat([train_master0,train_master1])
train_userupdateinfo=pd.concat([train_userupdateinfo0,train_userupdateinfo1])
train_loginfo=pd.concat([train_loginfo0,train_loginfo1])
#进行EDA和数据清理
#处理大量数据缺失的特征，根据是否缺失与target的直方图可见，缺失项与target的相关性也是比较明显的，因此对缺失项另外赋予一个值，由于后面会对数据进行独热(one-hot)编码，这些缺失项将会单独成为一个特征。
train_master,train_userupdateinfo,train_loginfo=EDA_and_data_cleaning(train_master,train_userupdateinfo,train_loginfo,path)
#对缺失值填充0，剔除标准差几乎为零的特征项
#把特征中的数值特征与分类特征分开，以便接下来对两类特征的分别处理
#绘制每种特征与target的散点图，然后手动剔除异常值，该步骤能够使训练模型的时候更加精准
#大部分数值特征的分布都集中在0附近，不适合接下来的建模，通过把数据对数化，使数据的分布趋向正态分布

train_master,train_userupdateinfo,train_loginfo=preprocessing_and_feature_engineering(train_master,train_userupdateinfo,train_loginfo,path,graph)
#提取重要特征结果存储在csv中
#处理地点数据，首先筛选出户籍省份和居住地省份中违约率高的地点，将其二值化，然后删除原始特征
#对于userupdate表中的数据，先将日期数据拆分开来，然后将其余信息转化为：用户信息更新的次数，用户信息更新的种类数，用户分几次更新了，用户借款成交与信息更新时间跨度
#对于LogInfo表将数据处理成以下几种特征1)累计登陆次数2)登陆时间的平均间隔3)最近一次的登陆时间距离成交时间差
#对于独热编码后的结果，再次进行相关度检验，清除部分相关性极低的变量
#由于飞桨与本地python中，pandas等包存在版本差异，部分代码无法表达原有意图
feature_selection(train_master,train_userupdateinfo,train_loginfo,path,graph)
#对数据直接进行xgboot建模
single_model_xgb(path)
#再次用lightgbm提取重要度较高的变量，再lgb建模
single_model_lgb(path)
#模型融合：xgboost adaboost randomforest
model_ensemble(path)