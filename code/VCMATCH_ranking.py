import json
import time
import math
import pandas as pd
import numpy as np
import warnings
import logging

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

import xgboost as xgb
import lightgbm as lgb
from torch.autograd import Variable

from VCMATCH_encoding_module import *

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
             'verbose': -1
             }
    print("LightGBM training & predicting")
    model = lgb.train(param, lgb.Dataset(
        data=X_train, label=y_train), num_boost_round=500)
    model.save_model('../data/lgboost_model.txt')
    predict = model.predict(X_test)
    return predict


# ============== CNN ==============
class CNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float)
        self.y = torch.tensor(np.array(y), dtype=torch.long)
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.y[idx]
        return data, label


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Net(nn.Module):
    def __init__(self, num_feature):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_feature, 32),
            nn.Linear(32, 8),
            nn.Linear(8, 2)
        )
        self.soft = nn.Softmax()

    def forward(self, input_):
        s1 = self.model(input_)
        out = self.soft(s1)
        return out


def cnn(X_train, y_train, X_test):
    lr = 0.001
    num_workers = 10
    alpha = 10
    batch_size = 10000
    num_epoches = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = FocalLoss1(class_num=2, alpha=torch.tensor([1, 100])) # changed from by me
    criterion = FocalLoss(class_num=2, alpha=torch.tensor([1, 100]))    # changed to by me

    train_dataset = CNNDataset(X_train, y_train)
    test_dataset = CNNDataset(X_test, pd.Series([1]*X_test.shape[0]))
    num_feature = X_train.shape[1]
    model = Net(num_feature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=False)

    print("CNN training & predicting")
    for epoch in range(num_epoches):
        model.train()
        predict = []
        t1 = time.time()
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            label_size = data.size()[0]
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time.time()
        logger.info('Epoch [{}/{}], Time {}s, Loss: {:.4f}, Lr:{:.4f}'.format(
            epoch + 1, num_epoches, int(t2 - t1), loss.item(), lr))
        torch.save(model.state_dict(),
                   '../data/cnn_20_{:02}.ckpt'.format(epoch))

    model.eval()
    with torch.no_grad():
        predict = []
        for i, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            pred = model(data)
            pred = pred.cpu().detach().numpy()
            predict.extend(pred)
        predict = np.array(predict)
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
    df1['rank'] = df1['level_0']+1
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
        idx = item2[item2.label == 1].index[0]+1
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
    for i in range(start, end+1):
        metric1, metric2 = get_score(df, 'rank_'+suffix, i)
        metric1_list.append(metric1)
        metric2_list.append(metric2)
    result['Manual_Efforts_'+suffix] = metric1_list
    result['Recall_'+suffix] = metric2_list
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
            list3.append(1/(math.log((idx+1), 2)))
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
            return l[1]+l[2]
        else:
            return l[1]+l[0]

    result['closest'] = result.apply(lambda row: get_closest(row, cols), axis=1)
    result['sum'] = 0
    # for column in columns:    # changed from by me
    for column in cols:      # changed to by me
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


# ========== 5-fold cross-validation ==========
df2 = pd.read_csv("../data/Dataset_150.csv")
repos = df2.repo.unique()
data_set2 = [pd.read_csv("../data/dataset/feature_{}.csv".format(repo)) for repo in repos]
df = pd.concat(data_set2)
df = df.reset_index(drop=True)
cvelist = df.cve.unique()
# kf = KFold(n_splits=5, shuffle=True)

# # save as pickle file
# with open('../data/model_kf.pkl', 'wb') as f:
#     pickle.dump(kf, f)

# load model_kf.pkl
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
vuln_cols = ['vuln_emb' + str(i) for i in range(32)]
cmt_cols = ['cmt_emb' + str(i) for i in range(32)]

result = df[['cve', 'commit', 'label']]
result.loc[:, 'prob_xgb'] = 0
result.loc[:, 'prob_lgb'] = 0
result.loc[:, 'prob_cnn'] = 0


for idx, (train_index, test_index) in enumerate(kf.split(cvelist)):
    cve_train = cvelist[train_index]
    isTrain = df.cve.apply(lambda item: item in cve_train)
    train = df[isTrain]
    test = df[isTrain==False]

    tmp_train = train[['cve', 'repo', 'commit', 'label']].copy()    # changed to by me
    tmp_test = test[['cve', 'repo', 'commit', 'label']].copy()      # changed to by me

    outpath = '../data/BERT-encode/'
    note = 'idx_'+str(idx)
    print("model.py {}".format(note))
    # BERT_encoding(tmp_train, tmp_test, note)

    train[vuln_cols] = readfile(outpath + 'vuln_embedding_train_' + note)
    train[cmt_cols] = readfile(outpath + 'commit_embedding_train_' + note)
    test[vuln_cols] = readfile(outpath + 'vuln_embedding_test_' + note)
    test[cmt_cols] = readfile(outpath + 'commit_embedding_test_' + note)

    # X_train = train[feature_cols + vuln_cols + cmt_cols]
    # y_train = train['label']
    # X_test = test[feature_cols + vuln_cols + cmt_cols]
    # y_test = test['label']
    X_train = train[feature_cols]
    y_train = train['label']
    X_test = test[feature_cols]
    y_test = test['label']

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

    # --- cnn
    cnn_predict = cnn(X_train, y_train, X_test)
    num4 = 0
    for item in test_index:
        for i in range(150):
            result['prob_cnn'][item * 150 + i] = cnn_predict[num4][1]
            num4 += 1


# prepare path
result_path = '../data/result_HandFeature/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

result.to_csv(result_path + "result.csv", index=False)
result = pd.read_csv(result_path + "result.csv")  # reset index

# save rank result
result['rank_xgb'] = get_rank(result, ['prob_xgb'])
result['rank_lgb'] = get_rank(result, ['prob_lgb'])
result['rank_cnn'] = get_rank(result, ['prob_cnn'])
result.to_csv(result_path + "rank_result.csv", index=False)

# save fusion-rank result
tmp_col1 = ['prob_xgb', 'prob_lgb']
result1 = fusion_avg(result, tmp_col1)
result1.to_csv(result_path + "rank_fusion_avg_result.csv", index=False)
tmp_col2 = ['rank_xgb', 'rank_lgb', 'rank_cnn']
result2 = fusion_voting(result, tmp_col2)
result2.to_csv(result_path + "rank_fusion_voting_result.csv", index=False)

# save metric result
result_metric = pd.DataFrame()
result_metric = get_full_score(result, 'fusion_avg', result_metric)
result_metric.to_csv(result_path + "metric_Recall_ME_avg.csv", index=False)
result_metric = get_full_score(result, 'fusion_voting', result_metric)
result_metric.to_csv(result_path + "metric_Recall_ME_voting.csv", index=False)

result3 = pd.DataFrame()
result3 = ndcg(result, result3)
result3.to_csv(result_path + "metric_NDCG.csv", index=False)
