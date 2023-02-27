
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
def model_ensemble(path):
    train = pd.read_csv(path + 'train.csv', encoding='utf-8')
    train=train.fillna(0)
    #print(train.isnull().any())
    train_all = train[0:28855]
    test_all = train[28855:58855]
    y_train = train_all.pop('target')
    test_all.pop('target')
    estimators = []
    estimators.append(('XGBClassifier', XGBClassifier(
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
        seed=27)))
    estimators.append(('AdaBoostClassifier', AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=81,
        algorithm="SAMME.R",
        learning_rate=0.5
    )))
    estimators.append(('RandomForestClassifier', RandomForestClassifier()))

    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(train_all, y_train)
    print(voting.score(train_all, y_train))
    pred=voting.predict(test_all)
    res = pd.DataFrame({"Idx": test_all.values[:, 0], "target": pred})
    res.to_csv('result.csv', encoding='utf-8', index=False)