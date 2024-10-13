import datetime
import glob
import json
import os
import gc
import git
import logging
from tqdm import tqdm
import shutil
from collections import Counter
import math
from util import *

with open('../data/token_IDF.txt', 'r') as fp:
    token_IDF = eval(fp.read())


def counter_to_dict(vunl_df_desc_token_counter):
    i = 0
    for item in vunl_df_desc_token_counter:
        item = item.replace('Counter', '')
        item = item.replace('(', '')
        item = item.replace(')', '')
        item = item.replace('{', '')
        item = item.replace('}', '')
        item = item.replace("'", '')
        item = item.replace('"', '')
        item = item.replace(' ', '')
        # print(item)
        # print(mess_df['mess_token_counter'][i])
        # vuln_df['desc_token_counter'][i] = dict(item)
        dic = {}
        if item == '':
            dic = {}
        elif item.find(',') < 0:
            key, value = item.split(':')
            dic[key] = int(value)
        else:
            for pair in item.split(','):
                key, value = pair.split(':')
                dic[key] = int(value)
        vunl_df_desc_token_counter[i] = dic
        i += 1
    return vunl_df_desc_token_counter


def weblinks_bug_issue_cve(weblinks, bug, issue, cve, row):
    issue_cnt = len(issue)
    web_cnt = len(weblinks)
    bug_cnt = len(bug)
    cve_cnt = len(cve)

    return issue_cnt, web_cnt, bug_cnt, cve_cnt


def feature_time(committime, cvetime):
    committime = datetime.datetime.strptime(committime, '%Y%m%d')
    cvetime = datetime.datetime.strptime(cvetime, '%Y%m%d')
    time_dis = abs((cvetime - committime).days)
    return time_dis


def file_match_func(filepaths, funcs, desc):
    files = [path.split('/')[-1] for path in filepaths]
    file_match = As_in_B(files, desc)
    filepath_match = As_in_B(filepaths, desc)
    func_match = As_in_B(funcs, desc)
    return file_match, filepath_match, func_match


def vuln_commit_token(reponame, commit, cwedesc_tokens, desc_tokens):
    vuln_tokens = union_list(desc_tokens, cwedesc_tokens)  # 合并列表

    with open('../data/gitcommit/{}/{}'.format(reponame, commit), 'r') as fp:
        commit_tokens = eval(fp.read())

    commit_tokens_set = set(commit_tokens)

    inter_token_total = inter_token(set(vuln_tokens), commit_tokens_set)
    inter_token_total_cnt = len(inter_token_total)
    inter_token_total_ratio = inter_token_total_cnt / len(vuln_tokens)

    inter_token_cwe = inter_token(set(cwedesc_tokens), commit_tokens_set)
    inter_token_cwe_cnt = len(inter_token_cwe)
    inter_token_cwe_ratio = inter_token_cwe_cnt / (1 + len(cwedesc_tokens))

    return vuln_tokens, commit_tokens, inter_token_total_cnt, inter_token_total_ratio, inter_token_cwe_cnt, inter_token_cwe_ratio


def get_vuln_idf(bug, links, cve, cves):
    cve_match = 0
    for item in cves:
        if item.lower() in cve.lower():
            cve_match = 1
            break

    bug_match = 0
    for link in links:
        if 'bug' in link or 'Bug' in link or 'bid' in link:
            for item in bug:
                if item in link:
                    bug_match = 1
                    break

    return cve_match, bug_match


def get_vuln_loc(nvd_items, commit_items):
    same_cnt = 0
    commit_items = set(commit_items)
    commit_items = list(commit_items)
    for commit_item in commit_items:
        for nvd_item in nvd_items:
            if nvd_item == commit_item:
                same_cnt += 1
                break
    same_ratio = same_cnt / (len(commit_items)+1)
    unrelated_cnt = len(commit_items) - same_cnt
    return same_cnt, same_ratio, unrelated_cnt


def get_vuln_type_relete(nvd_type, nvd_impact, commit_type, commit_impact, vuln_type_impact):# v_typ v_imp mes_typ mes_imp
    l1, l2, l3 = 0, 0, 0
    for nvd_item in nvd_type:
        for commit_item in commit_type:
            if nvd_item == commit_item:
                l1 += 1
            else:
                l3 += 1

    for nvd_item in nvd_type:
        for commit_item in commit_impact:
            if commit_item in vuln_type_impact.get(nvd_item):
                l2 += 1
            else:
                l3 += 1

    for commit_item in commit_type:
        for nvd_item in nvd_impact:
            if nvd_item in vuln_type_impact.get(commit_item):
                l2 += 1
            else:
                l3 += 1
    cnt = l1 + l2 + l3+1
    return l1/cnt, l2/cnt, (l3+1)/cnt

# c1 # nvd
# c2 # code


def get_vuln_desc_text(c1, c2):
    c3 = c1 and c2
    same_token = c3.keys()
    shared_num = len(same_token)
    shared_ratio = shared_num / (len(c1.keys())+1)
    c3_value = list(c3.values())
    if len(c3_value) == 0:
        c3_value = [0]
    return shared_num, shared_ratio, max(c3_value), sum(c3_value), np.mean(c3_value), np.var(c3_value)


def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]]
                    for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def get_feature(row, commit_info):  # row是Dataset_5000中1000行连续且repo相同的数据中的其中一行
    commit = row['commit']
    reponame = row['repo']
    weblinks, bug, issue, cve, datetime, filepaths, funcs, addcnt, delcnt = commit_info[commit]

    # [特征] CVE & Bug & Issue & Links 匹配
    # issue_cnt, web_cnt, bug_cnt, cve_cnt, \
    #     web_match_nvd_links, issue_match_nvd_links, bug_match_nvd_links, cve_match  \
    #     = weblinks_bug_issue_cve(weblinks, bug, issue, cve, row)

    issue_cnt, web_cnt, bug_cnt, cve_cnt, \
        = weblinks_bug_issue_cve(weblinks, bug, issue, cve, row)

    # [特征]  时间差值
    time_dis = feature_time(datetime, row['cvetime'])

    # [特征]  vuln 与 Commit tokens 重叠特征
    vuln_tokens, commit_tokens, inter_token_total_cnt, inter_token_total_ratio, inter_token_cwe_cnt, inter_token_cwe_ratio \
        = vuln_commit_token(reponame, commit, row['cwedesc'], row['desc_token'])

    # [特征] vuln 与 commit 的TFIDF向量的余弦相似度值
    c_vuln = Counter(vuln_tokens)
    c_commit = Counter(commit_tokens)
    len_vuln_tokens = len(vuln_tokens) + 1
    len_commit_tokens = len(commit_tokens) + 1
    vuln_tfidf = []
    commit_tfidf = []
    for token in token_IDF:
        vuln_tfidf.append(c_vuln[token]/len_vuln_tokens * token_IDF[token])
        commit_tfidf.append(
            c_commit[token]/len_commit_tokens * token_IDF[token])
    tfidf = cosine_similarity(vuln_tfidf, commit_tfidf)

    # return addcnt, delcnt, issue_cnt, web_cnt, bug_cnt, cve_cnt, \
    #     web_match_nvd_links, issue_match_nvd_links, bug_match_nvd_links, cve_match, \
    #     time_dis, inter_token_cwe_cnt, inter_token_cwe_ratio, tfidf

    return addcnt, delcnt, issue_cnt, web_cnt, bug_cnt, cve_cnt, time_dis, inter_token_cwe_cnt, inter_token_cwe_ratio, tfidf


dataset_df = pd.read_csv("../data/Dataset_150.csv")
dataset_df = reduce_mem_usage(dataset_df)   # 内存优化


vuln_df = pd.read_csv('../data/vuln_data.csv')
vuln_df['desc_token'] = vuln_df['desc_token'].apply(eval)
counter_to_dict(vuln_df['desc_token_counter'])
vuln_df['links'] = vuln_df['links'].apply(eval)
vuln_df['cwedesc'] = vuln_df['cwedesc'].apply(eval)
vuln_df['cvetime'] = vuln_df['cvetime'].astype(str)
vuln_df['functions'] = vuln_df['functions'].apply(eval)
vuln_df['files'] = vuln_df['files'].apply(eval)
vuln_df['filepaths'] = vuln_df['filepaths'].apply(eval)
vuln_df['vuln_type'] = vuln_df['vuln_type'].apply(eval)
vuln_df['vuln_impact'] = vuln_df['vuln_impact'].apply(eval)
vuln_df = reduce_mem_usage(vuln_df)

mess_df = pd.read_csv("../data/mess_data.csv")
mess_df['mess_bugs'] = mess_df['mess_bugs'].apply(eval)
mess_df['mess_cves'] = mess_df['mess_cves'].apply(eval)
mess_df['mess_type'] = mess_df['mess_type'].apply(eval)
mess_df['mess_impact'] = mess_df['mess_impact'].apply(eval)
counter_to_dict(mess_df['mess_token_counter'])
mess_df = reduce_mem_usage(mess_df)

dataset_df = dataset_df.merge(vuln_df, how='left', on='cve')
dataset_df = dataset_df.merge(mess_df, how='left', on='commit')

del vuln_df  # 删除
gc.collect()  # 内存清理

# ----------------------vuln_type_impact-------------------
with open("../data/vuln_type_impact.json", 'r') as f:
    vuln_type_impact = json.load(f)

repos = dataset_df.repo.unique()  # 去除重复元素，即得到repo的值域
for reponame in repos:
    dirpath = 'tmp/' + reponame
    # 生成tmp/reponame文件夹
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    tmp_df = dataset_df[dataset_df.repo == reponame]  # reponame对应的所有行
    gitpath = "../gitrepo/"
    repo = git.Repo(gitpath+reponame)
    commits = tmp_df.commit.unique()  # 去除重复元素，即得到commit的值域
    commit_info = readfile('../data/commit_info/{}_commit_info'.format(reponame))  # 读取reponame_commit_info文件

    total_cnt = tmp_df.shape[0]  # 读取行数
    each_cnt = 1000
    epoch = int((total_cnt+each_cnt-1)/each_cnt)
    logging.info('共有{}个epoch'.format(epoch))

    t1 = time.time()
    # batch process

    # 生成tmp/reponame/xxxx.csv文件
    for i in tqdm(range(epoch)):
        if os.path.exists(dirpath+'/{:04}.csv'.format(i)):
            continue
        df = tmp_df.iloc[i * each_cnt: min((i + 1) * each_cnt, total_cnt)]  # 从1000*i+1行开始的最多1000行数据

        df["addcnt"], df["delcnt"], df["issue_cnt"], df["web_cnt"], df["bug_cnt"], df["cve_cnt"],\
            df["time_dis"], df["inter_token_cwe_cnt"], df["inter_token_cwe_ratio"], df["vuln_commit_tfidf"] \
            = zip(*df.apply(lambda row: get_feature(row, commit_info), axis=1))  # 对每一行使用get_feature
        df['totalcnt'] = df["addcnt"] + df["delcnt"]

        df.drop(['desc', 'links', 'cwedesc', 'cvetime'], axis=1, inplace=True)
        df.drop(['mess_bugs', 'mess_cves', 'functions', 'filepaths',
                 'files', 'vuln_type', 'vuln_impact', 'mess_type', 'mess_impact',
                 'desc_token_counter', 'mess_token_counter'],
                axis=1, inplace=True)

# 删除后特征 cve repo true_commit commit label desc_token desc_token_counter functions files filepaths vuln_type
        # vuln_impact mess_bugs mess_cves mess_type mess_impact mess_token_counter
        df.to_csv(dirpath+'/{:04}.csv'.format(i), index=False)
    t2 = time.time()
    logging.info('{}共耗时：{} min'.format(reponame, (t2 - t1) / 60))
    gc.collect()

    files = glob.glob(dirpath + '/*.csv')  # 匹配tmp/reponame下的所有csv文件的路径
    m = {}
    for file in files:
        idx = int(re.search('([0-9]+).csv', file).group(1))  # 将xxxx.csv文件的路径存入m[xxxx]
        m[idx] = file

    l = []
    for i in range(epoch):
        tmp = pd.read_csv(m[i])
        l.append(tmp)

    data_df = pd.concat(l)  # 将上述处理好的csv文件合并
    data_df = data_df.reset_index(drop=True)

    shutil.rmtree(dirpath)  # 删除reponame文件夹

    logging.info("")

    # print('当前时间为', time.strftime("%H:%M:%S"))
    dirpath = 'tmp/' + reponame
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # print(reponame+"正在处理")

    code_df = pd.read_csv("../data/code_data/code_data_{}.csv".format(reponame))
    # 字典化
    code_df['code_files'] = code_df['code_files'] .apply(eval)
    code_df['code_funcs'] = code_df['code_funcs'].apply(eval)
    code_df['code_filepaths'] = code_df['code_filepaths'] .apply(eval)
    # code_df['code_token_counter'] = code_df['code_token_counter'].apply(eval)
    counter_to_dict(code_df['code_token_counter'])
    tmp_df = dataset_df[dataset_df.repo == reponame]
    tmp_df = tmp_df.merge(code_df, how='left', on='commit')
    commits = tmp_df.commit.unique()  # commit的值域

    total_cnt = tmp_df.shape[0]
    each_cnt = 2000
    epoch = int((total_cnt+each_cnt-1)/each_cnt)
    logging.info('共有{}个epoch'.format(epoch))

    t1 = time.time()
    for i in tqdm(range(epoch)):
        if os.path.exists(dirpath+'/{:04}.csv'.format(i)):
            continue
        df = tmp_df.iloc[i * each_cnt: min((i + 1) * each_cnt, total_cnt)]
        df['cve_match'], df['bug_match'] = zip(*df.apply(
            lambda row: get_vuln_idf(row['mess_bugs'], row['links'], row['cve'], row['mess_cves']), axis=1))
        df['filepath_same_cnt'], df['filepath_same_ratio'], df['filepath_unrelated_cnt'] = zip(*df.apply(
            lambda row: get_vuln_loc(row['filepaths'], row['code_filepaths']), axis=1))
        df['func_same_cnt'], df['func_same_ratio'], df['func_unrelated_cnt'] = zip(*df.apply(
            lambda row: get_vuln_loc(row['functions'], row['code_funcs']), axis=1))
        df['file_same_cnt'], df['file_same_ratio'], df['file_unrelated_cnt'] = zip(*df.apply(
            lambda row: get_vuln_loc(row['files'], row['code_files']), axis=1))
        df['vuln_type_1'], df['vuln_type_2'], df['vuln_type_3'] = zip(*df.apply(
            lambda row: get_vuln_type_relete(row['vuln_type'], row['vuln_impact'], row['mess_type'], row['mess_impact'], vuln_type_impact), axis=1))
        df['mess_shared_num'], df['mess_shared_ratio'], df['mess_max'], df['mess_sum'], df['mess_mean'], df['mess_var'] = zip(*df.apply(
            lambda row: get_vuln_desc_text(row['desc_token_counter'], row['mess_token_counter']), axis=1))
        df['code_shared_num'], df['code_shared_ratio'], df['code_max'], df['code_sum'], df['code_mean'], df['code_var'] = zip(*df.apply(
            lambda row: get_vuln_desc_text(row['desc_token_counter'], row['code_token_counter']), axis=1))

        df.drop(['desc', 'cwedesc', 'cvetime'], axis=1, inplace=True)

        df.drop(['mess_bugs', 'links', 'mess_cves', 'functions', 'code_funcs', 'filepaths', 'code_filepaths',
                 'files', 'code_files', 'vuln_type', 'vuln_impact', 'mess_type', 'mess_impact',
                 'desc_token_counter', 'mess_token_counter', 'code_token_counter'],
                axis=1, inplace=True)

        df.to_csv(dirpath+'/{:04}.csv'.format(i), index=False)
    t2 = time.time()
    logging.info('{}共耗时：{} min'.format(reponame, (t2 - t1) / 60))
    gc.collect()

    files = glob.glob(dirpath+'/*.csv')
    m = {}
    l = []
    for file in files:
        idx = int(re.search('([0-9]+).csv', file).group(1))
        m[idx] = file
    l = [pd.read_csv(m[i]) for i in range(epoch)]
    data_df2 = pd.concat(l)
    data_df2 = data_df2.reset_index(drop=True)
    shutil.rmtree(dirpath)

    tmp_columns = ['cve_match', 'bug_match', 'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',
                   'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                   'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',
                   'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                   'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                   'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']
    data_df[tmp_columns] = data_df2[tmp_columns]

    data_df.to_csv('../data/dataset/feature_{}.csv'.format(reponame), index=False)
