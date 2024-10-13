import json
import pickle
import time
import math
import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_log_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from encoding_module import *

warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ XGBoost ============
def xgboost(X_train, y_train, X_test):
    # prefix = "xgb_"
    param = {
        'max_depth': 5,
        'eta': 0.05,
        'verbosity': 1,
        'random_state': 2021,
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist'
    }

    def myFeval(preds, dtrain):
        labels = dtrain.get_label()
        return 'error', math.sqrt(mean_squared_log_error(preds, labels))

    print("XGBoost training & predicting")
    xgb_train = xgb.DMatrix(X_train, y_train)
    model = xgb.train(param, xgb_train, num_boost_round=500, feval=myFeval)
    model.save_model('../data/xgboost_model.json')
    predict = model.predict(xgb.DMatrix(X_test))
    return predict


# ============ LightGBM ============
def lightgbm(X_train, y_train, X_test):
    # prefix = "lgb_"
    param = {'device': 'gpu',
             'learning_rate': 0.04,
             'max_depth': 5,
             'verbose': -1,
             # 'objective': 'binary',
             # 'metric': 'binary_logloss',
             }
    print("LightGBM training & predicting")
    model = lgb.train(param, lgb.Dataset(
        data=X_train, label=y_train), num_boost_round=500)
    model.save_model('../data/lgboost_model.txt')
    predict = model.predict(X_test)
    return predict


# ============= metric =============
# sort data based on 'sortby' list, and then get the rank of each data
def get_rank(df1, sortby, ascending=False):
    gb = df1.groupby('cve')
    l = []
    for item1, item2 in gb:
        item2 = item2.reset_index()
        item2 = item2.sort_values(sortby + ['commit'], ascending=ascending)
        item2 = item2.reset_index(drop=True).reset_index()
        l.append(item2[['index', 'level_0']])

    df1 = pd.concat(l)
    df1['rank'] = df1['level_0'] + 1
    df1 = df1.sort_values(['index'], ascending=True).reset_index(drop=True)  #
    return df1['rank']


# get metric
def get_score(test, rankname='rank', N=10):
    cve_list = []
    cnt = 0
    total = []
    gb = test.groupby('cve')
    for item1, item2 in gb:
        item2 = item2.sort_values(
            [rankname], ascending=True).reset_index(drop=True)
        idx = item2[item2.label == 1].index[0] + 1
        if idx <= N:
            total.append(idx)
            cnt += 1
        else:
            total.append(N)
            cve_list.append(item1)
    return np.mean(total), cnt / len(total)


def get_full_score(df, suffix, result, start=1, end=10):
    metric1_list = []
    metric2_list = []
    for i in range(start, end + 1):
        metric1, metric2 = get_score(df, 'rank_' + suffix, i)
        metric1_list.append(metric1)
        metric2_list.append(metric2)
    result['Manual_Efforts_' + suffix] = metric1_list
    result['Recall_' + suffix] = metric2_list
    return result


def ndcg(result1, result2):
    gb = result1.groupby('cve')
    list1 = []
    list2 = []
    list3 = []
    for item1, item2 in gb:
        list1.append(item1)
        item2 = item2.reset_index(drop=True)
        idx = item2['rank_fusion_voting'][0]
        if idx == 1:
            list2.append(1)
        else:
            list2.append(0)
        if idx <= 5:
            list3.append(1 / (math.log((idx + 1), 2)))
        else:
            list3.append(0)

    result2['cve'] = list1
    result2['ndcg@1'] = list2
    result2['ndcg@5'] = list3
    return result2


# ========== model fusion ==========
def fusion_voting(result, cols, suffix=''):
    def get_closest(row, columns):
        l = [row[column] for column in columns]
        l.sort()
        if l[1] - l[0] >= l[2] - l[1]:
            return l[1] + l[2]
        else:
            return l[1] + l[0]

    result['closest'] = result.apply(lambda row: get_closest(row, cols), axis=1)
    result['sum'] = 0
    # for column in columns:    # changed from by me
    for column in cols:  # changed to by me
        result['sum'] = result['sum'] + result[column]
    result['last'] = result['sum'] - result['closest']
    result['rank_fusion_voting' + suffix] = get_rank(result, ['closest', 'last'], True)
    result.drop(['sum', 'closest', 'last'], axis=1)
    return result


def fusion_avg(result, cols):
    def get_avg(row, columns):
        return sum([row[column] for column in columns]) / 2

    result['fusion_avg'] = result.apply(lambda row: get_avg(row, cols), axis=1)
    result['rank_fusion_avg'] = get_rank(result, ['fusion_avg'], False)
    result.drop(['fusion_avg'], axis=1)
    return result


# # ========== 5-fold cross-validation ==========
df2 = pd.read_csv("../data/Dataset_150.csv")
repos = df2.repo.unique()
data_set2 = [pd.read_csv("../data/dataset/feature_{}.csv".format(repo)) for repo in repos]
df = pd.concat(data_set2)
df = df.reset_index(drop=True)
cvelist = df.cve.unique()
# kf = KFold(n_splits=5, shuffle=True, random_state=1)
#
# # save as pickle file
# with open('../data/model_kf.pkl', 'wb') as f:
#     pickle.dump(kf, f)

# load kf.pkl
with open('../data/model_kf.pkl', 'rb') as f:
    kf = pickle.load(f)

feature_cols = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                'cve_match', 'bug_match',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']   # 37
feature_cols_1 = ['issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                'cve_match', 'bug_match',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']   # 37
feature_cols_2 = ['addcnt', 'delcnt', 'totalcnt',
                'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']   # 37
feature_cols_3 = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                'cve_match', 'bug_match',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']   # 37
feature_cols_4 = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'time_dis', 'inter_token_cwe_cnt', 'inter_token_cwe_ratio', 'vuln_commit_tfidf',
                'cve_match', 'bug_match',
                'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']   # 37
feature_cols_5 = ['addcnt', 'delcnt', 'totalcnt', 'issue_cnt', 'web_cnt', 'bug_cnt', 'cve_cnt',
                'time_dis',
                'cve_match', 'bug_match',
                'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                'vuln_type_1', 'vuln_type_2', 'vuln_type_3'
                ]   # 37

Roberta_cols = ['Roberta_emb' + str(i) for i in range(32)]

result = df[['cve', 'commit', 'label']]
result.loc[:, 'prob_xgb'] = 0
result.loc[:, 'prob_lgb'] = 0
result.loc[:, 'prob_cat'] = 0


def readfile(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


for idx, (train_index, test_index) in enumerate(kf.split(cvelist)):
    cve_train = cvelist[train_index]
    isTrain = df.cve.apply(lambda item: item in cve_train)
    train = df[isTrain]
    test = df[isTrain==False]

    tmp_train = train[['cve', 'repo', 'commit', 'label']].copy()
    tmp_test = test[['cve', 'repo', 'commit', 'label']].copy()

    outpath = '../data/RoBERTa-encode-1than50/'
    note = 'idx_' + str(idx)
    print("model.py {}".format(note))
    # RoBERTa_encoding(tmp_train, tmp_test, note)

    train[Roberta_cols] = readfile(outpath + 'RoBERTa_embedding_train_' + note)
    test[Roberta_cols] = readfile(outpath + 'RoBERTa_embedding_test_' + note)

    X_train = train[feature_cols+ Roberta_cols]
    y_train = train['label']
    X_test = test[feature_cols+ Roberta_cols]
    y_test = test['label']
    # X_train = train[feature_cols]
    # y_train = train['label']
    # X_test = test[feature_cols]
    # y_test = test['label']
    # X_train = train[Roberta_cols]
    # y_train = train['label']
    # X_test = test[Roberta_cols]
    # y_test = test['label']
    # --- xgboost
    xgb_predict = xgboost(X_train, y_train, X_test)
    num2 = 0
    for item in test_index:
        for i in range(150):
            result['prob_xgb'][item * 150 + i] = xgb_predict[num2]
            num2 += 1

    # --- lightgbm
    lgb_predict = lightgbm(X_train, y_train, X_test)
    num3 = 0
    for item in test_index:
        for i in range(150):
            result['prob_lgb'][item * 150 + i] = lgb_predict[num3]
            num3 += 1

    # --- catboost
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=5,
        verbose=100,
    )
    model.fit(train_pool)
    cat_pred = model.predict(test_pool)

    num4 = 0
    for item in test_index:
        for i in range(150):
            result['prob_cat'][item * 150 + i] = cat_pred[num4]
            num4 += 1

# prepare path
result_path = '../data/result_RoBERTa_1than50/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


result.to_csv(result_path + "result.csv", index=False)
result = pd.read_csv(result_path + "result.csv")  # reset index

# save rank result
result['rank_xgb'] = get_rank(result, ['prob_xgb'])
result['rank_lgb'] = get_rank(result, ['prob_lgb'])
result['rank_cat'] = get_rank(result, ['prob_cat'])
result.to_csv(result_path + "rank_result.csv", index=False)

# result = pd.read_csv(result_path + "rank_result.csv")

# save fusion-rank result
tmp_col2 = ['rank_xgb', 'rank_lgb', 'rank_cat']
result2 = fusion_voting(result, tmp_col2)
result2.to_csv(result_path + "rank_fusion_voting_result.csv", index=False)

# save metric result
result_metric = pd.DataFrame()
# result_metric = get_full_score(result, 'fusion_avg', result_metric)
# result_metric.to_csv(result_path + "metric_Recall_ME_avg.csv", index=False)
result_metric = get_full_score(result, 'fusion_voting', result_metric)
result_metric.to_csv(result_path + "metric_Recall_ME_voting.csv", index=False)

result3 = pd.DataFrame()
result3 = ndcg(result, result3)
result3.to_csv(result_path + "metric_NDCG.csv", index=False)
