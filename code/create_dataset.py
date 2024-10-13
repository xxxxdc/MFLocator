import git
import numpy as np
import pandas as pd
import time


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    repos = df.repo.unique()
    repo_commit = dict()
    for reponame in repos:
        t1 = time.time()
        repo = git.Repo('../gitrepo/{}'.format(reponame))
        total_commits = [str(item) for item in repo.iter_commits()]
        commits = []
        for commit in total_commits:
            try:
                outputs = repo.git.diff(
                    commit + '~1', commit, ignore_blank_lines=True, ignore_space_at_eol=True).split('\n')
                # repo.git.diff(commit+'~1', commit)
                commits.append(commit)
            except:
                print('pass commit:{}'.format(commit))
                pass
        t2 = time.time()
        repo_commit[reponame] = commits

    with open('../data/repo_commit.txt', 'w') as fp:
        fp.write(str(repo_commit))

    # load repo_commit.txt
    with open('../data/repo_commit.txt', 'r') as fp:
        repo_commit = eval(fp.read())

    # create dataset
    neg_num = 149
    total_data = []
    for cve in df.groupby('cve'):
        cve_id = cve[0]
        reponame = list(cve[1].repo.unique())[0]
        pos_commit = list(cve[1].commit.unique())[0]
        neg_commits = []
        np.random.shuffle(repo_commit[reponame])
        commits = repo_commit[reponame]
        cnt = 0
        for commit in commits:
            if cnt >= neg_num:
                break
            if commit == pos_commit:
                continue
            else:
                neg_commits.append(commit)
                cnt += 1
        total_data.append([cve_id, reponame, pos_commit, neg_commits])

    dataset = []
    for item in total_data:
        dataset.append([item[0], item[1], item[2], item[2], 1])
        for commit in item[3]:
            dataset.append([item[0], item[1], item[2], commit, 0])

    df = pd.DataFrame(dataset, columns=['cve', 'repo', 'true_commit', 'commit', 'label'])
    df.to_csv('../data/Dataset_150.csv', index=False)
