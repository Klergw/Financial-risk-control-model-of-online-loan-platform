import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
def single_model_lgb(path):

    df_final = pd.read_csv(path + 'train.csv', encoding='utf-8')
    train=df_final[0:28855]
    train['sample_status']='train'
    test=df_final[28855:58855]
    test['sample_status'] = 'test'
    df_final= pd.concat([train,test],axis=0).reset_index(drop=True)
    df_final.shape

    # 用lightGBM筛选特征,
    # 这里训练10个模型,并对10个模型输出的特征重要性取平均,最后对特征重要性的值进行归一化
    # 以上将训练集和测试集合并是为了处理特征,现在再将两者划分开,用于模型训练
    # 将三万训练集划分成训练集和测试集,没有目标标签的3万样本作为预测集

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df_final[df_final.sample_status == 'train'].drop(['Idx', 'sample_status', 'target'], axis=1),
        df_final[df_final.sample_status == 'train']['target'],
        test_size=0.3,
        random_state=0)

    train_fea = np.array(X_train)
    test_fea = np.array(X_test)
    evaluate_fea = np.array(
        df_final[df_final.sample_status == 'test'].drop(['Idx', 'sample_status', 'target'], axis=1))

    # # reshape(-1,1转成一列
    train_label = np.array(y_train).reshape(-1, 1)
    test_label = np.array(y_test).reshape(-1, 1)
    evaluate_label = np.array(df_final[df_final.sample_status == 'test']['target']).reshape(-1, 1)

    fea_names = list(X_train.columns)
    feature_importance_values = np.zeros(len(fea_names))
    # 训练10个lightgbm，并对10个模型输出的feature_importances_取平均

    import lightgbm as lgb
    from lightgbm import plot_importance

    for i in np.arange(10):
        model = lgb.LGBMClassifier(n_estimators=1000,
                                   learning_rate=0.05,
                                   n_jobs=-1,
                                   verbose=-1)
        model.fit(train_fea, train_label,
                  eval_metric='auc',
                  eval_set=[(test_fea, test_label)],
                  early_stopping_rounds=100,
                  verbose=-1)
        feature_importance_values += model.feature_importances_ / 10

    # 将feature_importance_values存成临时表
    fea_imp_df1 = pd.DataFrame({'feature': fea_names,
                                'fea_importance': feature_importance_values})
    fea_imp_df1 = fea_imp_df1.sort_values('fea_importance', ascending=False).reset_index(drop=True)
    fea_imp_df1['norm_importance'] = fea_imp_df1['fea_importance'] / fea_imp_df1[
        'fea_importance'].sum()  # 特征重要性value的归一化
    fea_imp_df1['cum_importance'] = np.cumsum(fea_imp_df1['norm_importance'])  # 特征重要性value的累加值

    fea_imp_df1
    # 特征重要性可视化
    plt.figure(figsize=(16, 16))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplot(3, 1, 1)
    plt.title('特征重要性')
    sns.barplot(data=fea_imp_df1.iloc[:10, :], x='norm_importance', y='feature')

    plt.subplot(3, 1, 2)
    plt.title('特征重要性累加图')
    plt.xlabel('特征个数')
    plt.ylabel('cum_importance')
    plt.plot(list(range(1, len(fea_names) + 1)), fea_imp_df1['cum_importance'], 'r-')

    plt.subplot(3, 1, 3)
    plt.title('各个特征的归一化得分')
    plt.xlabel('特征')
    plt.ylabel('norm_importance')
    plt.plot(fea_imp_df1.feature, fea_imp_df1['norm_importance'], 'b*-')
    #plt.show()
    # 特征重要性可视化
    plt.figure(figsize=(16, 16))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplot(3, 1, 1)
    plt.title('特征重要性')
    sns.barplot(data=fea_imp_df1.iloc[:10, :], x='norm_importance', y='feature')

    plt.subplot(3, 1, 2)
    plt.title('特征重要性累加图')
    plt.xlabel('特征个数')
    plt.ylabel('cum_importance')
    plt.plot(list(range(1, len(fea_names) + 1)), fea_imp_df1['cum_importance'], 'r-')

    plt.subplot(3, 1, 3)
    plt.title('各个特征的归一化得分')
    plt.xlabel('特征')
    plt.ylabel('norm_importance')
    plt.plot(fea_imp_df1.feature, fea_imp_df1['norm_importance'], 'b*-')
    #plt.show()
    # 特征重要性可视化
    plt.figure(figsize=(16, 16))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.subplot(3, 1, 1)
    plt.title('特征重要性')
    sns.barplot(data=fea_imp_df1.iloc[:10, :], x='norm_importance', y='feature')

    plt.subplot(3, 1, 2)
    plt.title('特征重要性累加图')
    plt.xlabel('特征个数')
    plt.ylabel('cum_importance')
    plt.plot(list(range(1, len(fea_names) + 1)), fea_imp_df1['cum_importance'], 'r-')

    plt.subplot(3, 1, 3)
    plt.title('各个特征的归一化得分')
    plt.xlabel('特征')
    plt.ylabel('norm_importance')
    plt.plot(fea_imp_df1.feature, fea_imp_df1['norm_importance'], 'b*-')
    #plt.show()

    # 剔除特征重要性为0的变量
    zero_imp_col = list(fea_imp_df1[fea_imp_df1.fea_importance == 0].feature)
    fea_imp_df11 = fea_imp_df1[~(fea_imp_df1.feature.isin(zero_imp_col))]
    print('特征重要性为0的变量个数为 ：{}'.format(len(zero_imp_col)))
    print(zero_imp_col)

    # 剔除特征重要性比较弱的变量
    low_imp_col = list(fea_imp_df11[fea_imp_df11.cum_importance >= 0.99].feature)
    print('特征重要性比较弱的变量个数为：{}'.format(len(low_imp_col)))
    print(low_imp_col)

    # 删除特征重要性为0和比较弱的特征
    drop_imp_col = zero_imp_col + low_imp_col
    mydf_final_fea_selected = df_final.drop(drop_imp_col, axis=1)
    mydf_final_fea_selected.shape

    mydf_final_fea_selected.to_csv(
        path+'mydf_final_fea_selected.csv', encoding='gbk', index=False)

    # 将该数据集切分成训练集和测试集,并通过调参提高精度,然后使用精度最高的模型预测3万个样本的标签

    # 导入数据.用于建模
    df = pd.read_csv(path+'mydf_final_fea_selected.csv', encoding='gbk')

    x_data = df[df.sample_status == 'train'].drop(['Idx', 'sample_status', 'target'], axis=1)
    y_data = df[df.sample_status == 'train']['target']

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.2)

    # 训练模型
    lgb_sklearn = lgb.LGBMClassifier(random_state=0).fit(x_train, y_train)

    # # 预测测试集的样本
    lgb_sklearn_pre = lgb_sklearn.predict_proba(x_test)

    ###计算roc和auc
    from sklearn.metrics import roc_curve, auc
    def acu_curve(y, prob):
        #  y真实,
        #  prob预测
        fpr, tpr, threshold = roc_curve(y, prob)  ###计算真阳性率(真正率)和假阳性率(假正率)
        roc_auc = auc(fpr, tpr)  ###计算auc的值

        plt.figure()
        lw = 2
        plt.figure(figsize=(12, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC')
        plt.legend(loc="lower right")

        plt.show()

    acu_curve(y_test, lgb_sklearn_pre[:, 1])
