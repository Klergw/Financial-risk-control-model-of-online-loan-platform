import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def EDA_and_data_cleaning(train_master,train_userupdateinfo,train_loginfo,path):
    train_master0=train_master[0:30000]
    #借款人信息
    train_master.head() ## 借款人的一些信息
    train_master.info()
    train_master.shape
    #借款成交时间 , 修改内容 ,修改时间
    train_userupdateinfo.head(20)  ###借款成交时间 , 修改内容 ,修改时间
    train_userupdateinfo.info()
    train_userupdateinfo.shape
    #借款成交时间 ,操作代码 ,操作类别 ,登陆时间
    train_loginfo.head(20)  ##借款成交时间 ,操作代码 ,操作类别 ,登陆时间
    train_loginfo.shape
    train_loginfo.isnull().sum().sort_values(ascending=False).head(10)  # 缺失值统计
    #用户登录信息和用户更新信息没有缺失值，不用处理
    list(train_master.columns)
    ## 加载微软雅黑中文字体
    from matplotlib.font_manager import FontProperties
    myfont = FontProperties(fname=r"yahei.ttf", size=12)
    n_null_rate = train_master0.isnull().sum().sort_values(ascending=False) / 30000
    n_null_rate.head(20)
    ## 去掉缺失比例接近百分之百的字段
    train_master.drop(['WeblogInfo_1', 'WeblogInfo_3'], axis=1, inplace=True)

    ## 处理UserInfo_12缺失
    train_master0['UserInfo_12'].unique()
    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_UserInfo_12_not = train_master0.target[train_master0.UserInfo_12.isnull()].value_counts()
    target_UserInfo_12_ = train_master0.target[train_master0.UserInfo_12.notnull()].value_counts()
    df_UserInfo_12 = pd.DataFrame({'missing': target_UserInfo_12_not, 'not_missing': target_UserInfo_12_})
    df_UserInfo_12
    df_UserInfo_12.plot(kind='bar', stacked=True)
    plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    plt.xlabel(u'有无', fontproperties=myfont)
    plt.ylabel(u'违约情况', fontproperties=myfont)
    #plt.show()
    train_master.loc[(train_master.UserInfo_12.isnull(), 'UserInfo_12')] = 2.0
    # train_master['UserInfo_11'].fillna(2.0)
    # train_master['UserInfo_12'] =train_master['UserInfo_12'].astype(np.int32)
    train_master['UserInfo_12'].dtypes
    train_master['UserInfo_12'].unique()

    ## 处理UserInfo_11缺失
    train_master0['UserInfo_11'].unique()
    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_UserInfo_11_not = train_master0.target[train_master0.UserInfo_11.isnull()].value_counts()
    target_UserInfo_11_ = train_master0.target[train_master0.UserInfo_11.notnull()].value_counts()
    df_UserInfo_11 = pd.DataFrame({'no_have': target_UserInfo_11_not, 'have': target_UserInfo_11_})
    df_UserInfo_11
    df_UserInfo_11.plot(kind='bar', stacked=True)
    plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    plt.xlabel(u'有无', fontproperties=myfont)
    plt.ylabel(u'违约情况', fontproperties=myfont)
    #plt.show()

    # train_master['UserInfo_11'] =train_master['UserInfo_11'].astype(str)
    train_master.loc[(train_master.UserInfo_11.isnull(), 'UserInfo_11')] = 2.0
    train_master['UserInfo_11'].unique()

    ## 处理UserInfo_13缺失
    train_master['UserInfo_13'].unique()
    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_UserInfo_13_not = train_master0.target[train_master0.UserInfo_13.isnull()].value_counts()
    target_UserInfo_13_ = train_master0.target[train_master0.UserInfo_13.notnull()].value_counts()
    df_UserInfo_13 = pd.DataFrame({'no_have': target_UserInfo_13_not, 'have': target_UserInfo_13_})
    df_UserInfo_13
    df_UserInfo_13.plot(kind='bar', stacked=True)
    plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    plt.xlabel(u'有无', fontproperties=myfont)
    plt.ylabel(u'违约情况', fontproperties=myfont)
    #plt.show()

    # train_master['UserInfo_13'] =train_master['UserInfo_13'].astype(str)
    train_master.loc[(train_master.UserInfo_13.isnull(), 'UserInfo_13')] = 2.0
    train_master['UserInfo_13'].unique()

    ## 处理WeblogInfo_20 缺失
    train_master['WeblogInfo_20'].unique()
    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_WeblogInfo_20_not = train_master0.target[train_master0.WeblogInfo_20.isnull()].value_counts()
    target_WeblogInfo_20_ = train_master0.target[train_master0.WeblogInfo_20.notnull()].value_counts()
    df_WeblogInfo_20 = pd.DataFrame({'no_have': target_WeblogInfo_20_not, 'have': target_WeblogInfo_20_})
    df_WeblogInfo_20
    df_WeblogInfo_20.plot(kind='bar', stacked=True)
    plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    plt.xlabel(u'有无', fontproperties=myfont)
    plt.ylabel(u'违约情况', fontproperties=myfont)
    #plt.show()

    # train_master['WeblogInfo_20'] =train_master['WeblogInfo_20'].astype(str)
    train_master.loc[(train_master.WeblogInfo_20.isnull(), 'WeblogInfo_20')] = u'不详'
    train_master['WeblogInfo_20'].unique()

    train_master['WeblogInfo_19'].unique()

    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_WeblogInfo_19_not = train_master0.target[train_master0.WeblogInfo_19.isnull()].value_counts()
    target_WeblogInfo_19_ = train_master0.target[train_master0.WeblogInfo_19.notnull()].value_counts()
    df_WeblogInfo_19 = pd.DataFrame({'no_have': target_WeblogInfo_19_not, 'have': target_WeblogInfo_19_})
    df_WeblogInfo_19

    # df_WeblogInfo_19.plot(kind='bar', stacked=True)
    # plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    # plt.xlabel(u'有无', fontproperties=myfont)
    # plt.ylabel(u'违约情况', fontproperties=myfont)
    # plt.show()

    # train_master['WeblogInfo_19'] =train_master['WeblogInfo_19'].astype(str)
    train_master.loc[(train_master.WeblogInfo_19.isnull(), 'WeblogInfo_19')] = u'不详'
    train_master['WeblogInfo_19'].unique()

    ## 处理WeblogInfo_21 缺失
    train_master['WeblogInfo_21'].unique()
    # fig = plt.figure()
    # fig.set(alpha=0.2)
    target_WeblogInfo_21_not = train_master0.target[train_master0.WeblogInfo_21.isnull()].value_counts()
    target_WeblogInfo_21_ = train_master0.target[train_master0.WeblogInfo_21.notnull()].value_counts()
    df_WeblogInfo_21 = pd.DataFrame({'no_have': target_WeblogInfo_21_not, 'have': target_WeblogInfo_21_})
    df_WeblogInfo_21

    df_WeblogInfo_21.plot(kind='bar', stacked=True)
    plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
    plt.xlabel(u'有无', fontproperties=myfont)
    plt.ylabel(u'违约情况', fontproperties=myfont)
    #plt.show()

    # train_master['WeblogInfo_21'] =train_master['WeblogInfo_21'].astype(str)
    train_master.loc[(train_master.WeblogInfo_21.isnull(), 'WeblogInfo_21')] = '0'
    train_master['WeblogInfo_21'].unique()

    return train_master,train_userupdateinfo,train_loginfo