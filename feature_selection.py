import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def feature_selection(train_master,train_userupdateinfo,train_loginfo,path,graph):
    train_master0 = train_master[0:28855]
    import re
    train_master['UserInfo_8'] = [a[:-1] if a.find('市') != -1 else a[:] for a in train_master['UserInfo_8']]

    # 清理后非重复计数减小
    train_master['UserInfo_8'].nunique()

    # 1)省份特征————————推测可能一个是籍贯省份，一个是居住省份
    # 首先看看各省份好坏样本的分布占比
    def get_badrate(df, col):
        '''
        根据某个变量计算违约率
        '''
        group = train_master0.groupby(col)
        df = pd.DataFrame()
        df['total'] = group.target.count()
        df['bad'] = group.target.sum()
        df['badrate'] = round(df['bad'] / df['total'], 4) * 100  # 百分比形式
        return df.sort_values('badrate', ascending=False)

    # 户籍省份的违约率计算
    province_original = get_badrate(train_master, 'UserInfo_19')
    province_original

    province_current = get_badrate(train_master, 'UserInfo_7')
    province_current
    # 分别对户籍省份和居住省份排名前五的省份进行二值化
    # 户籍省份的二值化
    train_master['is_tianjin_UserInfo_19'] = train_master.apply(lambda x: 1 if x.UserInfo_19 == '天津市' else 0, axis=1)
    train_master['is_shandong_UserInfo_19'] = train_master.apply(lambda x: 1 if x.UserInfo_19 == '山东省' else 0, axis=1)
    train_master['is_jilin_UserInfo_19'] = train_master.apply(lambda x: 1 if x.UserInfo_19 == '吉林省' else 0, axis=1)
    train_master['is_heilongjiang_UserInfo_19'] = train_master.apply(lambda x: 1 if x.UserInfo_19 == '黑龙江省' else 0,
                                                                     axis=1)
    train_master['is_hunan_UserInfo_19'] = train_master.apply(lambda x: 1 if x.UserInfo_19 == '湖南省' else 0, axis=1)

    # 居住省份的二值化
    train_master['is_tianjin_UserInfo_7'] = train_master.apply(lambda x: 1 if x.UserInfo_7 == '天津' else 0, axis=1)
    train_master['is_shandong_UserInfo_7'] = train_master.apply(lambda x: 1 if x.UserInfo_7 == '山东' else 0, axis=1)
    train_master['is_sichuan_UserInfo_7'] = train_master.apply(lambda x: 1 if x.UserInfo_7 == '四川' else 0, axis=1)
    train_master['is_hainan_UserInfo_7'] = train_master.apply(lambda x: 1 if x.UserInfo_7 == '海南' else 0, axis=1)
    train_master['is_hunan_UserInfo_7'] = train_master.apply(lambda x: 1 if x.UserInfo_7 == '湖南' else 0, axis=1)

    # 户籍省份和居住地省份不一致的特征衍生
    print(train_master.UserInfo_19.unique())
    print('\n')
    print(train_master.UserInfo_7.unique())

    # 首先将两者改成相同的形式
    UserInfo_19_change = []
    for i in train_master.UserInfo_19:
        if i in ('内蒙古自治区', '黑龙江省'):
            j = i[:3]
        else:
            j = i[:2]
        UserInfo_19_change.append(j)
    print(np.unique(UserInfo_19_change))

    # 判断UserInfo_7和UserInfo_19是否一致
    is_same_province = []
    for i, j in zip(train_master.UserInfo_7, UserInfo_19_change):
        if i == j:
            a = 1
        else:
            a = 0
        is_same_province.append(a)
    train_master['is_same_province'] = is_same_province

    # 2)城市特征
    # 原数据中有四个城市特征,推测为用户常登陆的IP地址城市
    # 特征衍生思路:
    # 一,通过xgboost挑选重要的城市,进行二值化
    # 二,由四个城市特征的非重复计数衍生生成登陆IP地址的变更次数

    # 根据xgboost变量重要性的输出对城市作二值化衍生
    df_Master_temp = train_master[['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20', 'target']]
    df_Master_temp.head()
    area_list = []
    # 将四个城市特征都进行哑变量处理
    for col in df_Master_temp:
        dummy_df = pd.get_dummies(df_Master_temp[col])
        dummy_df = pd.concat([dummy_df, df_Master_temp['target']], axis=1)
        area_list.append(dummy_df)
    df_area1 = area_list[0]
    df_area2 = area_list[1]
    df_area3 = area_list[2]
    df_area4 = area_list[3]

    df_area1
    # 使用xgboost筛选出重要的城市
    #from pandas import MultiIndex, Int16Dtype
    #from xgboost.sklearn import XGBClassifier
    #from xgboost import plot_importance

    # 注意,这里需要把合并后的没有目标标签的行数据删除
    # df_area1[~(df_area1['target'].isnull())]
    '''
    if (graph == True):
        x_area1 = df_area1[~(df_area1['target'].isnull())].drop(['target'], axis=1)
        y_area1 = df_area1[~(df_area1['target'].isnull())]['target']
        x_area2 = df_area2[~(df_area2['target'].isnull())].drop(['target'], axis=1)
        y_area2 = df_area2[~(df_area2['target'].isnull())]['target']
        x_area3 = df_area3[~(df_area3['target'].isnull())].drop(['target'], axis=1)
        y_area3 = df_area3[~(df_area3['target'].isnull())]['target']
        x_area4 = df_area4[~(df_area4['target'].isnull())].drop(['target'], axis=1)
        y_area4 = df_area4[~(df_area4['target'].isnull())]['target']

        xg_area1 = XGBClassifier(random_state=0).fit(x_area1, y_area1)
        xg_area2 = XGBClassifier(random_state=0).fit(x_area2, y_area2)
        xg_area3 = XGBClassifier(random_state=0).fit(x_area3, y_area3)
        xg_area4 = XGBClassifier(random_state=0).fit(x_area4, y_area4)

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        plot_importance(xg_area1, ax=ax1, max_num_features=10, height=0.4)
        plot_importance(xg_area2, ax=ax2, max_num_features=10, height=0.4)
        plot_importance(xg_area3, ax=ax3, max_num_features=10, height=0.4)
        plot_importance(xg_area4, ax=ax4, max_num_features=10, height=0.4)
'''
    # 将特征重要性排名前三的城市进行二值化：
    train_master['is_zibo_UserInfo_2'] = train_master.apply(lambda x: 1 if x.UserInfo_2 == '淄博' else 0, axis=1)
    train_master['is_chengdu_UserInfo_2'] = train_master.apply(lambda x: 1 if x.UserInfo_2 == '成都' else 0, axis=1)
    train_master['is_yantai_UserInfo_2'] = train_master.apply(lambda x: 1 if x.UserInfo_2 == '烟台' else 0, axis=1)

    train_master['is_zibo_UserInfo_4'] = train_master.apply(lambda x: 1 if x.UserInfo_4 == '淄博' else 0, axis=1)
    train_master['is_qingdao_UserInfo_4'] = train_master.apply(lambda x: 1 if x.UserInfo_4 == '青岛' else 0, axis=1)
    train_master['is_shantou_UserInfo_4'] = train_master.apply(lambda x: 1 if x.UserInfo_4 == '汕头' else 0, axis=1)

    train_master['is_zibo_UserInfo_8'] = train_master.apply(lambda x: 1 if x.UserInfo_8 == '淄博' else 0, axis=1)
    train_master['is_chengdu_UserInfo_8'] = train_master.apply(lambda x: 1 if x.UserInfo_8 == '成都' else 0, axis=1)
    train_master['is_heze_UserInfo_8'] = train_master.apply(lambda x: 1 if x.UserInfo_8 == '菏泽' else 0, axis=1)

    train_master['is_ziboshi_UserInfo_20'] = train_master.apply(lambda x: 1 if x.UserInfo_20 == '淄博市' else 0,
                                                                axis=1)
    train_master['is_chengdushi_UserInfo_20'] = train_master.apply(lambda x: 1 if x.UserInfo_20 == '成都市' else 0,
                                                                   axis=1)
    train_master['is_sanmenxiashi_UserInfo_20'] = train_master.apply(lambda x: 1 if x.UserInfo_20 == '三门峡市' else 0,
                                                                     axis=1)

    # 特征衍生-IP地址变更次数特征
    train_master['UserInfo_20'] = [a[:-1] if a.find('市') != -1 else i[:] for a in train_master.UserInfo_20]
    city_df = train_master[['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']]

    city_change_cnt = []
    for i in range(city_df.shape[0]):
        a = list(city_df.iloc[i])
        city_count = len(set(a))
        city_change_cnt.append(city_count)
    train_master['city_count_cnt'] = city_change_cnt
    '''
    # 3)运营商种类少,直接将其转换成哑变量
    print(train_master.UserInfo_9.value_counts())
    print(set(train_master.UserInfo_9))
    train_master['UserInfo_9'] = train_master.UserInfo_9.replace({'中国联通 ': 'china_unicom',
                                                                  '中国联通': 'china_unicom',
                                                                  '中国移动': 'china_mobile',
                                                                  '中国移动 ': 'china_mobile',
                                                                  '中国电信': 'china_telecom',
                                                                  '中国电信 ': 'china_telecom',
                                                                  '不详': 'operator_unknown'

                                                                  })

    operator_dummy = pd.get_dummies(train_master.UserInfo_9)
    train_master = pd.concat([train_master, operator_dummy], axis=1)
    
    # 删除原变量
    train_master = train_master.drop(['UserInfo_9'], axis=1)'''
    train_master = train_master.drop(
        ['UserInfo_19', 'UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_20'], axis=1)

    # 看看还剩下哪些类型变量要处理
    train_master.dtypes.value_counts()
    train_master.select_dtypes(include='object')
    ## 去掉大小写
    train_userupdateinfo['UserupdateInfo1'] = train_userupdateinfo['UserupdateInfo1'].apply(lambda x: x.lower())
    train_userupdateinfo.to_csv(path + 'train_userupdateinfo.csv', index=False, encoding='utf-8')

    ## 借款日期离散化
    # 把月、日、单独拎出来，放到3列中
    train_master['month'] = pd.DatetimeIndex(train_master.ListingInfo).month
    train_master['day'] = pd.DatetimeIndex(train_master.ListingInfo).day
    train_master['day'].head()
    train_master.drop(['ListingInfo'], axis=1, inplace=True)
    train_master['target'] = train_master['target'].astype(str)
    train_master.to_csv(path + 'train_master.csv', index=False, encoding='utf-8')

    from collections import defaultdict
    import datetime as dt
    ##  userupdateinfo表
    userupdate_info_number = defaultdict(list)  ### 用户信息更新的次数
    userupdate_info_category = defaultdict(set)  ###用户信息更新的种类数
    userupdate_info_times = defaultdict(list)  ### 用户分几次更新了
    userupdate_info_date = defaultdict(list)  #### 用户借款成交与信息更新时间跨度

    with open(path + 'train_userupdateinfo.csv', 'r') as f:
        f.readline()
        for line in f.readlines():
            cols = line.strip().split(",")  ### cols 是list结果
            userupdate_info_date[cols[0]].append(cols[1])
            userupdate_info_number[cols[0]].append(cols[2])
            userupdate_info_category[cols[0]].add(cols[2])
            userupdate_info_times[cols[0]].append(cols[3])
        print(u'提取信息完成')

    userupdate_info_number_ = defaultdict(int)  ### 用户信息更新的次数
    userupdate_info_category_ = defaultdict(int)  ### 用户信息更新的种类数
    userupdate_info_times_ = defaultdict(int)  ### 用户分几次更新了
    userupdate_info_date_ = defaultdict(int)  #### 用户借款成交与信息更新时间跨度

    for key in userupdate_info_date.keys():
        userupdate_info_times_[key] = len(set(userupdate_info_times[key]))
        delta_date = dt.datetime.strptime(userupdate_info_date[key][0], '%Y/%m/%d') - dt.datetime.strptime(
            list(set(userupdate_info_times[key]))[0], '%Y/%m/%d')
        userupdate_info_date_[key] = abs(delta_date.days)
        userupdate_info_number_[key] = len(userupdate_info_number[key])
        userupdate_info_category_[key] = len(userupdate_info_category[key])

    print('信息处理完成')

    ## 建立一个DataFrame
    Idx_ = list(userupdate_info_date_.keys())  #### list
    numbers_ = list(userupdate_info_number_.values())
    categorys_ = list(userupdate_info_category_.values())
    times_ = list(userupdate_info_times_.values())
    dates_ = list(userupdate_info_date_.values())
    userupdate_df = pd.DataFrame(
        {'Idx': Idx_, 'numbers': numbers_, 'categorys': categorys_, 'times': times_, 'dates': dates_})
    userupdate_df.head()
    userupdate_df.to_csv(path + 'userupdate_df.csv', index=False, encoding='utf-8')

    # LogInfo表

    # 衍生的变量有
    # 1)累计登陆次数
    # 2)登陆时间的平均间隔
    # 3)最近一次的登陆时间距离成交时间差

    # 1)累计登陆次数
    log_cnt = train_loginfo.groupby('Idx', as_index=False).LogInfo3.count().rename(
        columns={'LogInfo3': 'log_cnt'})
    log_cnt.head(10)

    # 2)最近一次的登陆时间距离成交时间差

    # 最近一次的登录时间距离当前时间差
    train_loginfo['Listinginfo1'] = pd.to_datetime(train_loginfo.Listinginfo1)
    train_loginfo['LogInfo3'] = pd.to_datetime(train_loginfo.LogInfo3)
    time_log_span = train_loginfo.groupby('Idx', as_index=False).agg({'Listinginfo1': np.max,
                                                                      'LogInfo3': np.max})
    time_log_span.head()

    time_log_span['log_timespan'] = time_log_span['Listinginfo1'] - time_log_span['LogInfo3']
    time_log_span['log_timespan'] = time_log_span['log_timespan'].map(lambda x: str(x))

    time_log_span['log_timespan'] = time_log_span['log_timespan'].map(lambda x: int(x[:x.find('d')]))
    time_log_span = time_log_span[['Idx', 'log_timespan']]
    time_log_span.head()

    log_info = pd.merge(log_cnt, time_log_span, how='left', on='Idx')
    log_info.head()

    log_info.to_csv(path + 'log_info_df.csv', index=False, encoding='utf-8')

    train_master = pd.read_csv(path + 'train_master.csv', encoding='utf-8')
    train_userupdateinfo = pd.read_csv(path + 'userupdate_df.csv', encoding='utf-8')
    train_loginfo = pd.read_csv(path + 'log_info_df.csv', encoding='utf-8')
    print(train_master.shape)
    print(train_userupdateinfo.shape)
    print(train_loginfo.shape)

    train_all = pd.merge(train_master, train_userupdateinfo, how='left', on='Idx')
    train_all = pd.merge(train_all, train_loginfo, how='left', on='Idx')
    print(train_all.shape)
    train_all.head()
    # test_all=pd.merge(test_master, test_userupdateinfo, how='left', on='Idx')
    # train_all = pd.merge(train_all, train_loginfo, how='left', on='Idx')
    train_all.isnull().sum().sort_values(ascending=False).head(10)
    # test_all.isnull().sum().sort_values(ascending=False).head(10)

    train_all = pd.merge(train_master, train_userupdateinfo, how='left', on='Idx')
    train_all = pd.merge(train_all, train_loginfo, how='left', on='Idx')
    print(train_all.shape)
    train_all.head()
    train_all.isnull().sum().sort_values(ascending=False).head(10)
    #对最后的数据进行独热编码，将200条特征变成4000多条，极大增加了数据的计算量和存储空间。通过计算每一种特征与target的相关性，然后剔除相关系数极低的特征，能够加快建模速度，且剔除前后对于模型预测能力没有显著影响。
    ## 对数值型特征进行scaling
    import warnings
    warnings.filterwarnings("ignore")
    train_all['Idx'] = train_all['Idx'].astype(np.int64)
    # test_all['Idx'] = test_all['Idx'].astype(np.int64)
    train_all['target'] = train_all['target'].astype(np.int64)
    print(train_all.shape)
    train_all = pd.get_dummies(train_all)
    train_all.head()
    print(train_all.shape)
    train_all.to_csv(path + 'train.csv', encoding='utf-8', index=False)
    train = pd.read_csv(path + 'train.csv', encoding='utf-8')

    train_master1 = train[0:28855]
    train_corr = np.abs(train_master1.corr()['target']).sort_values(ascending=False)
    train_corr.head(20)
    train_corr.to_csv(path + 'train_corr.csv', encoding='utf-8', index=True)
