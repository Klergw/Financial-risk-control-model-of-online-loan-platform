import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocessing_and_feature_engineering(train_master,train_userupdateinfo,train_loginfo,path,graph):
    train_master0 = train_master[0:30000]
    ## 其余缺失值很少的就用均值或众数填充
    len(train_master['UserInfo_2'].value_counts())  ## 城市地理位置
    len(train_master['UserInfo_4'].value_counts())  ## 城市地理位置
    len(train_master['UserInfo_8'].value_counts())  ## 城市地理位置
    len(train_master['UserInfo_9'].unique())  ## 城市地理位置
    len(train_master['UserInfo_20'].value_counts())  ## 城市地理位置
    len(train_master['UserInfo_7'].unique())  ## 省份地理位置
    len(train_master['UserInfo_19'].unique())  ## 省份地理位置

    # 如果选择以0填充，下述部分就维持现状，如果选择中位数/众数填充，就把下述的部分注释掉
    train_master.loc[(train_master.UserInfo_2.isnull(), 'UserInfo_2')] = '0'
    train_master.loc[(train_master.UserInfo_4.isnull(), 'UserInfo_4')] = '0'
    train_master.loc[(train_master.UserInfo_8.isnull(), 'UserInfo_8')] = '0'
    train_master.loc[(train_master.UserInfo_9.isnull(), 'UserInfo_9')] = '0'
    train_master.loc[(train_master.UserInfo_20.isnull(), 'UserInfo_20')] = '0'
    train_master.loc[(train_master.UserInfo_7.isnull(), 'UserInfo_7')] = '0'
    train_master.loc[(train_master.UserInfo_19.isnull(), 'UserInfo_19')] = '0'
    train_master.head()
    y_train = train_master0['target'].values

    ## 剔除标准差几乎为零的特征项
    feature_std = train_master.std().sort_values(ascending=True)
    feature_std.head(20)
    train_master.drop(['WeblogInfo_10', 'WeblogInfo_49'], axis=1, inplace=True)
    train_master['Idx'] = train_master['Idx'].astype(np.int32)

    for i in range(25):
        name = 'UserInfo_' + str(i)
        try:
            print(train_master[name].head())
        except:
            pass

    train_master['UserInfo_8'].head(20)

    ratio_threshold = 0.5
    binarized_features = []
    binarized_features_most_freq_value = []

    # 不同period的third_party_feature均值汇总在一起，结果并不好，故取消
    # third_party_features = []
    for f in train_master.columns:
        if f in ['target']:
            continue

        #     if 'ThirdParty_Info_Period' in f:
        #         third_party_features.append(f)
        #         continue

        not_null_sum = (train_master[f].notnull()).sum()
        most_count = pd.value_counts(train_master[f], ascending=False).iloc[0]
        most_value = pd.value_counts(train_master[f], ascending=False).index[0]
        ratio = most_count / not_null_sum

        if ratio > ratio_threshold:
            binarized_features.append(f)
            binarized_features_most_freq_value.append(most_value)

    numerical_features = [f for f in train_master.select_dtypes(exclude=['object']).columns
                          if f not in (['Idx', 'target'])
                          and f not in binarized_features]
    #                       and 'ThirdParty_Info_Period' not in f]
    categorical_features = [f for f in train_master.select_dtypes(include=["object"]).columns
                            if f not in (['Idx', 'target'])
                            and f not in binarized_features]
    #                         and 'ThirdParty_Info_Period' not in f]
    print(numerical_features)
    print(categorical_features)
    for i in range(len(binarized_features)):
        f = binarized_features[i]
        most_value = binarized_features_most_freq_value[i]
        train_master['b_' + f] = 1
        train_master.loc[train_master[f] == most_value, 'b_' + f] = 0
        train_master.drop([f], axis=1, inplace=True)

    feature_unique_count = []
    for f in numerical_features:
        feature_unique_count.append((np.count_nonzero(train_master[f].unique()), f))

    # print(sorted(feature_unique_count))

    for c, f in feature_unique_count:
        if c <= 10:
            print('{} moved from numerical to categorical'.format(f))
            numerical_features.remove(f)
            categorical_features.append(f)

    import seaborn as sns
    if (graph == True):
        melt = pd.melt(train_master0, id_vars=['target'], value_vars=[f for f in numerical_features])
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")

    test_master = train_master[30000:60000]
    train_master = train_master[0:30000]
    print('{} lines before drop'.format(train_master.shape[0]))

    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_1 > 250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period6_2 > 400].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_2 > 250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period6_3 > 2000].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_3 > 1250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period6_4 > 1500].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_4 > 1250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 400)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_7 > 2000)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_6 > 1500)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_5 > 1000) & (train_master.target == 0)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1500)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_8 > 1000) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period6_16 > 2000000) & (train_master.target == 0)].index,
        inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period6_14 > 1000000) & (train_master.target == 0)].index,
        inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_12 > 60)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 120) & (train_master.target == 0)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_11 > 20) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period6_13 > 200000)].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period6_13 > 150000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period6_15 > 40000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period6_17 > 130000) & (train_master.target == 0)].index,
        inplace=True)

    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_1 > 500].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_2 > 500].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 3000) & (train_master.target == 0)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_3 > 2000)].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_5 > 500].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_4 > 2000) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_6 > 700].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_6 > 300) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_7 > 4000)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_8 > 800)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period5_11 > 200)].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_13 > 200000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_14 > 150000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_15 > 75000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_16 > 180000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period5_17 > 150000].index, inplace=True)

    # go above

    train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_1 > 400)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_2 > 350)].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_3 > 1500)].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_4 > 1600].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_4 > 1250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_5 > 500].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_6 > 800].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period4_6 > 400) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_8 > 1000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_13 > 250000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_14 > 200000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_15 > 70000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_16 > 210000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period4_17 > 160000].index, inplace=True)

    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_1 > 400].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_2 > 380].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_3 > 1750].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_4 > 1750].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_4 > 1250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_5 > 600].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_6 > 800].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_6 > 400) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period3_7 > 1600) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_8 > 1000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_13 > 300000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_14 > 200000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_15 > 80000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_16 > 300000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period3_17 > 150000].index, inplace=True)

    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_1 > 400].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_1 > 300) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_2 > 400].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_2 > 300) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_3 > 1800].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_3 > 1500) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_4 > 1500].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_5 > 580].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_6 > 800].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_6 > 400) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_7 > 2100].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period2_8 > 700) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_11 > 120].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_13 > 300000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_14 > 170000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_15 > 80000].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period2_15 > 50000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_16 > 300000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period2_17 > 150000].index, inplace=True)

    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_1 > 350].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_1 > 200) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_2 > 300].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_2 > 190) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_3 > 1500].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_4 > 1250].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_5 > 400].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_6 > 500].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_6 > 250) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_7 > 1800].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_8 > 720].index, inplace=True)
    train_master.drop(train_master[(train_master.ThirdParty_Info_Period1_8 > 600) & (train_master.target == 1)].index,
                      inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_11 > 100].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_13 > 200000].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period1_13 > 140000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_14 > 150000].index, inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_15 > 70000].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period1_15 > 30000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_16 > 200000].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period1_16 > 100000) & (train_master.target == 1)].index,
        inplace=True)
    train_master.drop(train_master[train_master.ThirdParty_Info_Period1_17 > 100000].index, inplace=True)
    train_master.drop(
        train_master[(train_master.ThirdParty_Info_Period1_17 > 80000) & (train_master.target == 1)].index,
        inplace=True)

    train_master.drop(train_master[train_master.WeblogInfo_4 > 40].index, inplace=True)
    train_master.drop(train_master[train_master.WeblogInfo_6 > 40].index, inplace=True)
    train_master.drop(train_master[train_master.WeblogInfo_7 > 150].index, inplace=True)
    train_master.drop(train_master[train_master.WeblogInfo_16 > 50].index, inplace=True)
    train_master.drop(train_master[(train_master.WeblogInfo_16 > 25) & (train_master.target == 1)].index, inplace=True)
    train_master.drop(train_master[train_master.WeblogInfo_17 > 100].index, inplace=True)
    train_master.drop(train_master[(train_master.WeblogInfo_17 > 80) & (train_master.target == 1)].index, inplace=True)
    train_master.drop(train_master[train_master.UserInfo_18 < 10].index, inplace=True)

    print('{} lines after drop'.format(train_master.shape[0]))
    train_master = pd.concat([train_master, test_master])
    #以下运行时间过长
    '''
    # melt = pd.melt(train_master, id_vars=['target'], value_vars = [f for f in numerical_features if f != 'Idx'])
    if (graph == True):
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.distplot, "value")

    # train_master_log = train_master.copy()
    numerical_features_log = [f for f in numerical_features if f not in ['Idx']]

    for f in numerical_features_log:
        train_master[f + '_log'] = np.log1p(train_master[f])
        train_master.drop([f], axis=1, inplace=True)

    from math import inf

    (train_master == -inf).sum().sum()
    train_master.replace(-inf, -1, inplace=True)

    # log后的密度图，应该分布靠近正态分布了
    if (graph == True):
        melt = pd.melt(train_master, id_vars=['target'], value_vars=[f + '_log' for f in numerical_features])
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.distplot, "value")

    # log后的分布图，看是否有log后的outlier
    if (graph == True):
        g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
        g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")

    if (graph == True):
        melt = pd.melt(train_master, id_vars=['target'], value_vars=[f for f in categorical_features])
        g = sns.FacetGrid(melt, col='variable', col_wrap=4, sharex=False, sharey=False)
        g.map(sns.countplot, 'value', palette="muted")

'''
    return train_master, train_userupdateinfo, train_loginfo

