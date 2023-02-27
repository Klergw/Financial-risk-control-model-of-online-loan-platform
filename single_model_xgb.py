import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
import matplotlib.pylab as plt
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams


def modelfit(alg, dtrain, y_train, test, dtest=None, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values[:, 1:], label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # 建模
    alg.fit(dtrain.values[:, 1:], y_train, eval_metric='auc')
    alg.save_model('model.pkl')
    # 对训练集预测
    dtrain_predictions = alg.predict(dtrain.values[:, 1:])
    dtrain_predprob = alg.predict_proba(dtrain.values[:, 1:])[:, 1]
    # test_prediction=alg.predict(test)
    dtrain_predictions1 = alg.predict(test.values[:, 1:])
    #dtrain_predprob1 = alg.predict_proba(test.values[:, 1:])[:, 1]
    res = pd.DataFrame({"Idx": test.values[:, 0], "target": dtrain_predictions1})
    res.to_csv('result.csv', encoding='utf-8', index=False)
    # 输出模型的一些结果
    print(cvresult.shape[0])
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp.head(25))
    print(feat_imp.shape)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
def single_model_xgb(path):
    rcParams['figure.figsize'] = 12, 4

    train = pd.read_csv(path + 'train.csv', encoding='utf-8')
    #train_corr = pd.read_csv(path + 'train_corr.csv', encoding='utf-8')
    #train_corr.tail(1000)
    #train_corr_head = train_corr.tail(1000)
    # print(list(target_corr_head))
    #train.drop(list(train_corr_head['Unnamed: 0']), axis=1, inplace=True)

    train_all = train[0:28855]
    test_all = train[28855:58855]
    print(train_all.shape)
    print(test_all.shape)
    y_train = train_all.pop('target')
    test_all.pop('target')
    xgb1 = XGBClassifier(
        learning_rate=0.05,
        n_estimators=65,  # 树的个数
        max_depth=25,  # 树深度
        min_child_weight=8,
        gamma=0,  # 惩罚项系数
        subsample=0.5,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=23,
        seed=27
    )

    modelfit(xgb1, train_all, y_train, test_all)

