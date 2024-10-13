import os
import gc
import git
import math
import string
import logging
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
from util import *

# Global variable
gitpath = "../gitrepo/"
tokenpath = "../data/tokens/"
gitlogpath = "../data/gitlog/"
gitcommit_path = '../data/gitcommit/'
gitrepo_path = '../gitrepo/'

df = pd.read_csv("../data/data.csv")
repos = df.repo.unique()        # unique: return unique values
df_dataset = pd.read_csv("../data/Dataset_150.csv")
df_dataset = reduce_mem_usage(df_dataset)   # optimize memory usage

stopword_list = stopwords.words('english') + list(string.punctuation)


# ================= prepare path =================
def prepare_token_path():
    if not os.path.exists(tokenpath):
        os.makedirs(tokenpath)
    if not os.path.exists(gitlogpath):
        os.makedirs(gitlogpath)
    if not os.path.exists(gitcommit_path):
        os.makedirs(gitcommit_path)


prepare_token_path()


# ========== get commit-related token ==========
# single commit tokens
def get_commit_tokens(input):
    reponame, commit = input
    repo = git.Repo(gitrepo_path + reponame)

    # mess part
    temp_commit = repo.commit(commit)
    mess = temp_commit.message.replace('\r\n', ' ').replace('\n', ' ')
    mess = to_token(mess, None, stopword_list)

    # diff_code part
    filepaths, funcs,  codes = [], [], []
    outputs = repo.git.diff(commit + '~1', commit, ignore_blank_lines=True, ignore_space_at_eol=True).split('\n')
    for line in outputs:
        if re.match(r'^diff\s+--git', line):            # diff --git
            filepath = line.split(' ')[-1].strip()[2:]
            filepaths.extend(to_token(filepath))
        elif line.startswith('index'):                  # index
            continue
        elif line.startswith('@@ '):                    # @@ @@ func(
            funcname = line.split('@@')[-1].strip()
            funcname = funcs_preprocess(funcname)
            funcs.extend(to_token(funcname))
        elif line.startswith('++') or line.startswith('--'):    # +++ or ---
            continue
        elif line.startswith('+') or line.startswith('-'):      # + code or - code
            line_tokens = to_token(line[1:], None, stopword_list)
            codes.extend(line_tokens)
        elif line.strip() == "":                                #
            continue
        else:                                                   # code
            line_tokens = to_token(line, None, stopword_list)
            codes.extend(line_tokens)

    total_token = union_list(mess, filepaths, funcs, codes)
    return total_token


def multi_process_get_commit_tokens(repo, commits):
    repo_tokens = []
    for commit in commits:
        repo_tokens.append(get_commit_tokens((repo, commit)))
    repo_tokens_set = set()
    for item in repo_tokens:
        repo_tokens_set.update(item)
    return list(repo_tokens_set)


for reponame in repos:
    print(reponame + 'is being processed....')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    commits = df_dataset[df_dataset.repo == reponame].commit.unique()  # all commit_id in Dataset_150.csv that belong to this repo
    repo_tokens = multi_process_get_commit_tokens(reponame, commits)
    with open(tokenpath+'tokens_{}.txt'.format(reponame), 'w+') as fp:
        fp.write(str(repo_tokens))

paths = [tokenpath + 'tokens_{}.txt'.format(reponame) for reponame in repos]
commit_tokens = set()
for path in tqdm(paths):
    with open(path, 'r') as fp:
        token = eval(fp.read())     # execute an expression and return a python object
        commit_tokens.update(token)

with open(tokenpath + 'tokens_commit.txt', 'w') as fp:
    fp.write(str(commit_tokens))


# ====== get vulnerability-related token ======
# tokenize the text(using multiprocess)
def multi_process_line(lines, pool_num=5):
    res = []
    for line in lines:
        res.append(to_token(line, None, stopword_list))
    ret = set()
    for item in res:
        ret.update(item)
    return list(ret)


def get_tokens(path):
    df_vuln = pd.read_csv(path)
    df_vuln = df_vuln.loc[:, ['cwedesc', 'desc']]
    df_vuln['cwedesc'] = df_vuln['cwedesc'].apply(eval)
    df_vuln['cwedesc'] = df_vuln['cwedesc'].apply(join_list_to_string)

    lines = df_vuln.apply(lambda row: ','.join(row.astype(str)), axis=1).tolist()
    return multi_process_line(lines)


vuln_token = set()
vuln_token.update(get_tokens("../data/vuln_data.csv"))
with open(tokenpath + '/tokens_vuln.txt', 'w+') as fp:
    fp.write(str(vuln_token))


# ============ get useful token ============
with open(tokenpath + 'tokens_commit.txt', 'r') as fp:
    commit_tokens = eval(fp.read())
with open(tokenpath + 'tokens_vuln.txt', 'r') as fp:
    vuln_token = eval(fp.read())

use_tokens = vuln_token & commit_tokens
with open(tokenpath + 'tokens_useful.txt', 'w+') as fp:
    fp.write(str(use_tokens))


# ================  load useful token ================
with open(tokenpath + 'tokens_useful.txt', 'r') as fp:
    tokens = eval(fp.read())


# ==========  get vuln-related useful token ==========
vuln_df = pd.read_csv("../data/vuln_data.csv")
vuln_df['cwedesc'] = vuln_df['cwedesc'].apply(eval)
vuln_df['cwedesc'] = vuln_df['cwedesc'].apply(join_list_to_string)
vuln_df['cwedesc'] = vuln_df['cwedesc'].apply(lambda item: to_token(item, tokens))
vuln_df['desc_token'] = vuln_df['desc'].apply(lambda item: to_token(item, tokens))
vuln_df['total'] = vuln_df['cwedesc'] + vuln_df['desc_token']
vuln_df.to_csv("../data/vuln_data.csv", index=False)


# ===== get commit mess and diff_code useful token =====
# single commit tokens
def get_commit_token(input):
    reponame, commit = input
    repo = git.Repo(gitrepo_path + reponame)

    # mess part
    temp_commit = repo.commit(commit)
    mess = temp_commit.message.replace('\r\n', ' ').replace('\n', ' ')
    mess = to_token(mess, tokens)

    # diff_code part
    filepaths, funcs,  codes = [], [], []
    outputs = repo.git.diff(commit + '~1', commit, ignore_blank_lines=True, ignore_space_at_eol=True).split('\n')
    if len(outputs) >= 1000:
        outputs = outputs[:1000]
    for line in outputs:
        if re.match(r'^diff\s+--git', line):            # diff --git
            filepath = line.split(' ')[-1].strip()[2:]
            filepaths.extend(to_token(filepath, tokens))
        elif line.startswith('index'):                  # index
            continue
        elif line.startswith('@@ '):                    # @@ @@ func(
            funcname = line.split('@@')[-1].strip()
            funcname = funcs_preprocess(funcname)
            funcs.extend(to_token(funcname, tokens))
        elif line.startswith('++') or line.startswith('--'):    # +++ or ---
            continue
        elif line.startswith('+') or line.startswith('-'):      # + code or - code
            line_tokens = to_token(line[1:], tokens)
            codes.extend(line_tokens)
        elif line.strip() == "":                                #
            continue
        else:                                                   # code
            line_tokens = to_token(line, tokens)
            codes.extend(line_tokens)

    total_token = union_list(mess, filepaths, funcs, codes)
    with open(gitcommit_path + '{}/{}'.format(reponame, commit), 'w') as fp:
        fp.write(str(total_token))

    return None


def multi_process_get_commit_token(repo, commits):
    for commit in tqdm(commits, ncols=80):
        get_commit_token((repo, commit))


for reponame in repos:
    print(reponame + 'is being processed....')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    repo_gitcommit_path = gitcommit_path + reponame
    if not os.path.exists(repo_gitcommit_path):
        os.makedirs(repo_gitcommit_path)

    commits = df_dataset[df_dataset.repo == reponame].commit.unique()  # all commit_id in Dataset_5000.csv that belong to this repo
    multi_process_get_commit_token(reponame, commits)


# ===================== IDF =====================
# ------------------- commit --------------------
def get_commit_token_df(filepaths):
    token_dict = dict()
    for token in tokens:
        token_dict[token] = 0

    for file in tqdm(filepaths, ncols=80):
        with open(file, 'r') as fp:
            commit_token = set(eval(fp.read()))
        for item in commit_token:
            token_dict[item] += 1
        del commit_token
        gc.collect()
    return token_dict


files = glob('../data/gitcommit/*/*')
token_dict = get_commit_token_df(files)


# ------------------- vuln --------------------
def get_vuln_token_df(vuln_token):
    token_dict = {}
    for token in tokens:
        token_dict[token] = 0

    for token in tqdm(vuln_token, ncols=80):
        token = set(token)
        for item in token:
            token_dict[item] += 1
    return token_dict


df = pd.read_csv('../data/vuln_data.csv')
df['total'] = df['total'].apply(eval)
vuln_tokens = list(df['total'])

result = get_vuln_token_df(vuln_tokens)
for token in result.keys():
    token_dict[token] += result[token]


# ------------------- IDF --------------------
for item in token_dict.keys():
    token_dict[item] = np.log((len(files) + len(vuln_tokens)) / (token_dict[item] + 1))

with open('../data/token_IDF.txt', 'w') as fp:
    fp.write(str(token_dict))
