from collections import OrderedDict

import git
import os
import logging

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from util import *
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel, AutoModel)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AutoModel
from transformers import logging as lg
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk.data
import re
import os

tokenizer_RoBERTa = RobertaTokenizer.from_pretrained('../pretrained-model/roberta-large')
stemmer = nltk.stem.SnowballStemmer('english')
lg.set_verbosity_error()
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     # added by me
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda")
n_gpu = 1


class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        config = RobertaConfig.from_pretrained('../pretrained-model/roberta-large')
        # config4Code = RobertaConfig.from_pretrained('microsoft/codebert-base')
        config.num_labels = 2
        self.Roberta = AutoModel.from_pretrained('../pretrained-model/roberta-large', config=config)
        # self.CodeBERT = AutoModel.from_pretrained('microsoft/codebert-base',config=config4Code)
        # self.text_hidden_size = text_hidden_size
        # self.code_hidden_size = code_hidden_size
        # self.num_class = num_class
        for param in self.Roberta.parameters():
            param.requires_grad = True
        # for param in self.CodeBERT.parameters():
        #     param.requires_grad = True
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 32)
        self.linear4 = nn.Linear(32, 2)

    def forward(self, input1, labels):
        text_output = self.Roberta(input1, attention_mask=input1.ne(1))[1]  # [batch_size, hiddensize]
        logits = self.linear2(self.linear1(text_output))
        logits = self.linear3(logits)
        logits = self.linear4(logits)
        prob = torch.softmax(logits, -1)  # -1 means cal for each row
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


def savefile(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def readfile(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', text)


def RemoveTag(str1, key):  # change any word in 'key-[0-9]*' to ' '
    if type(str1) != str:
        str1 = str(str1)
    keys = key.split("/")
    patterns = [k + '-[0-9]*' for k in keys]
    pattern = ""
    for ip in range(len(patterns)):
        if ip == 0:
            pattern = patterns[ip]
        else:
            pattern = pattern + "|" + patterns[ip]
    return re.sub(pattern, ' ', str1)


def RemoveHttp(str1):
    if type(str1) != str:
        str1 = str(str1)
    httpPattern = '[a-zA-z]+://[^\s]*'
    return re.sub(httpPattern, ' ', str1)


def RemoveGit(str1):
    if type(str1) != str:
        str1 = str(str1)
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str1)


def textProcess(text):
    key = 'full'    # for issue_text or commit_text or commit_code
    final = []
    # remove
    text = RemoveHttp(text)
    text = RemoveTag(text, key)
    text = RemoveGit(text)
    sentences = tokenizer_RoBERTa.tokenize(text)  # divide into sentence
    for sentence in sentences:
        if len(final) >= 600:
            break
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)  # divide into word
        word_tokens = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
        for word in word_tokens:
            if word in stopwords.words('english'):
                continue
            else:
                final.append(str(stemmer.stem(word)))
    if len(final) == 0:
        text = ' '
    else:
        text = ' '.join(final)
    return text


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,

                 label,
                 issue_key,
                 commit_sha,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.issue_key = issue_key,
        self.commit_sha = commit_sha,


def convert_examples_to_features(row, tokenizer, commitType):  # commit_text/commit_code
    max_seq_length = 512
    # issue_text = textProcess(row['Issue_Text'])
    # issue_text = textProcess(row['desc'])
    # commit_text = textProcess(row[commitType])
    # issue_token = tokenizer.tokenize(issue_text)
    # commit_token = tokenizer.tokenize(commit_text)
    issue_token = eval(row['desc'])
    commit_token = eval(row[commitType])
    if len(issue_token) + len(commit_token) > max_seq_length - 3:
        if len(issue_token) > (max_seq_length - 3) / 2 and len(commit_token) > (max_seq_length - 3) / 2:
            issue_token = issue_token[:int((max_seq_length - 3) / 2)]
            commit_token = commit_token[:max_seq_length - 3 - len(issue_token)]
        elif len(issue_token) > (max_seq_length - 3) / 2:
            issue_token = issue_token[:max_seq_length - 3 - len(commit_token)]
        elif len(commit_token) > (max_seq_length - 3) / 2:
            commit_token = commit_token[:max_seq_length - 3 - len(issue_token)]
    combined_token = [tokenizer.cls_token] + issue_token + [tokenizer.sep_token] + commit_token + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    # padding
    if len(combined_ids) < max_seq_length:
        padding_length = max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(combined_token, combined_ids, row['label'], row['cve'], row['commit'])


class TextDataset_RoBERTa(Dataset):
    def __init__(self, df, tokenizer=tokenizer_RoBERTa):
        self.text_examples = []
        self.code_examples = []
        # if 'TRAIN' in file_path:
        #     # under sample
        #     # df_link = MySubSampler(pd.read_csv(file_path), args.seed)
        #     df_link = MySubSampler(df, 1)
        # else:
        #     df_link = pd.df
        # token + id + label
        for i_row, row in tqdm(df.iterrows(), total=df.shape[0]):
            # self.text_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Text'))
            # self.code_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Code'))
            self.text_examples.append(convert_examples_to_features(row, tokenizer, 'mess'))
            self.code_examples.append(convert_examples_to_features(row, tokenizer, 'commit_code'))
        assert len(self.text_examples) == len(self.code_examples), 'ErrorLength'

    def __len__(self):
        return len(self.text_examples)

    def __getitem__(self, i):
        return (torch.tensor(self.text_examples[i].input_ids),
                torch.tensor(self.code_examples[i].input_ids),
                torch.tensor(self.text_examples[i].label))


def add_desc(df, filepath='../data/cve_desc.csv'):
    desc = pd.read_csv(filepath, encoding='latin1')
    df = df.merge(desc, on='cve', how='left')
    return df


def add_mess(df, gitdir='../gitrepo/'):
    def get_commit_message(reponame, commit):
        gitrepo = git.Repo(gitdir + reponame)
        temp_commit = gitrepo.commit(commit)
        mess = temp_commit.message
        return mess

    df['mess'] = df.apply(
        lambda row: get_commit_message(row['repo'], row['commit']), axis=1)
    return df


def prepare_RoBERTa_encoding():
    if not os.path.exists('../data/RoBERTa-encode/RoBERTa_Embedding.csv'):
        df = pd.read_csv('../data/Dataset_150.csv')
        df = df[['cve', 'repo', 'commit']]
        df = add_desc(df)
        df = add_mess(df)   # load bert tokenizer
        desc = {'desc': []}
        list1 = []
        commit_code = {'commit_code': []}
        list2 = []
        commit = {'mess': []}
        list3 = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            issue_text = textProcess(row['desc'])
            code_text = textProcess(row['commit_code'])
            commit_text = textProcess(row['mess'])
            issue_token = tokenizer_RoBERTa.tokenize(issue_text)
            commit_token = tokenizer_RoBERTa.tokenize(commit_text)
            code_token = tokenizer_RoBERTa.tokenize(code_text)
            # use RoBERTa tokenizer
            list1.append(issue_token)
            list2.append(code_token)
            list3.append(commit_token)
        desc['desc'] = list1
        commit_code['commit_code'] = list2
        commit['commit'] = list3
        df['desc'] = pd.DataFrame(desc)['desc']
        df['commit_code'] = pd.DataFrame(commit_code)["commit_code"]
        df['mess'] = pd.DataFrame(commit)['mess']
        df.to_csv('../data/RoBERTa-encode/RoBERTa_Embedding.csv', index=False)


def create_RoBERTa_encode_dataset(train_df, test_df, trainDatasetPath='../data/RoBERTa-encode/RoBERTa_enc_train',
                                  testDatasetPath='../data/RoBERTa-encode/RoBERTa_enc_test', note='0'):
    prepare_RoBERTa_encoding()
    df = pd.read_csv('../data/RoBERTa-encode/RoBERTa_Embedding.csv')
    train_df = pd.merge(left=train_df, right=df, on=['cve', 'repo', 'commit'])
    test_df = pd.merge(left=test_df, right=df, on=['cve', 'repo', 'commit'])

    print('train_df')
    trainDataset = TextDataset_RoBERTa(train_df)
    print('test_df')
    testDataset = TextDataset_RoBERTa(test_df)

    savefile(trainDataset, trainDatasetPath)

    positive_samples = train_df[train_df['label'] == 1]
    negative_samples = train_df[train_df['label'] == 0]

    balanced_negative_samples = pd.DataFrame()  # save selected negative samples
    for cve_id, group in negative_samples.groupby('cve'):
        if len(group) > 10:
            sampled_group = group.sample(n=10, random_state=42)
        else:
            sampled_group = group
        balanced_negative_samples = pd.concat([balanced_negative_samples, sampled_group])

    train_df = pd.concat([positive_samples, balanced_negative_samples])

    trainDataset_10 = TextDataset_RoBERTa(train_df)

    savefile(trainDataset_10, '../data/RoBERTa-encode/RoBERTa_enc_train_' + note)
    savefile(testDataset, testDatasetPath)


def train_enc_RoBERTa(trainDatasetPath,
              testDatasetPath,
              criterion=None,
              optimizer=None,
              num_epochs=10,
              batch_size=8,
              shuffle=False,
              num_workers=1,
              weight_decay=0.0,
              adam_epsilon=1e-8,
              learning_rate=1e-5,
              max_grad_norm=0.1,
            note='0'):
    trainDataset = readfile(trainDatasetPath)
    train_sampler = RandomSampler(trainDataset)

    trainDataLoader = DataLoader(trainDataset, sampler=train_sampler,
                                  batch_size=batch_size, num_workers=4, pin_memory=True)

    model = Roberta().to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    max_steps = len(trainDataLoader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)

    model.zero_grad()
    model.train()
    for epoch in range(num_epochs):
        t1 = time.time()
        loss_sum = 0
        bar = tqdm(trainDataLoader, total=len(trainDataLoader))
        losses = []
        for i, (data1, data2, label) in enumerate(bar):
            data1 = data1.to(device)
            # data2 = data2.to(device)
            label = label.to(device)

            loss, logits = model(data1, label)
            # loss = criterion(pred, label)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        t2 = time.time()

        logging.info('Epoch [{:2}/{:2}], Loss: {:.4f}, Time: {:4}s'.format(epoch + 1, num_epochs, loss.item(), int(t2 - t1)))

        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), '../data/RoBERTa-encode/RoBERTa_{}_epoch_{}_{}.ckpt'.format(note, num_epochs, epoch))


def RoBERTa_evaluate(model, testDataset):

    eval_dataset = testDataset
    seed = 1

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=16, num_workers=4, pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = 16")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        text_inputs = batch[0].to(device)
        code_inputs = batch[1].to(device)
        label = batch[2].to(device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, code_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    print('Predictions', preds[:25])
    print('Labels:', labels[:25])

    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    return result


def get_RoBERTa_embedding(model,
                  dataset_path,
                  batch_size=8,
                  shuffle=False,
                  num_workers=1,
                  note='data',
                  outpath='../data/RoBERTa-encode/'):
    dataset = readfile(dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    model.eval()
    # data1_embedding = []
    # data2_embedding = []
    out_embedding = []
    with torch.no_grad():
        bar = tqdm(dataloader, total=len(dataloader))
        for i, (data1, data2, label) in enumerate(bar):
            data1 = data1.to(device)
            out1 = model.Roberta(data1, attention_mask=data1.ne(1))[1]
            out = model.linear3(model.linear2(model.linear1(out1)))
            out_embedding.extend(out.cpu().numpy())
    out_embedding = np.array(out_embedding)

    savefile(out_embedding, outpath + 'RoBERTa_embedding_' + note)


def RoBERTa_encoding(train_df, test_df, note):
    if not os.path.exists('../data/RoBERTa-encode'):
        os.makedirs('../data/RoBERTa-encode')

    trainDatasetPath = '../data/RoBERTa-encode/RoBERTa_enc_trainALL_' + note
    testDatasetPath = '../data/RoBERTa-encode/RoBERTa_enc_test_' + note

    print("Create dataset {}".format(str(note)))
    logging.info("Create dataset")
    create_RoBERTa_encode_dataset(train_df, test_df, trainDatasetPath, testDatasetPath, note=note)

    print("Train encoding module {}".format(str(note)))
    logging.info("Train encoding module")
    train_enc_RoBERTa(trainDatasetPath='../data/RoBERTa-encode/RoBERTa_enc_train_' + note,
                      testDatasetPath=testDatasetPath, note=note)  # RoBERTa fine-tune
    print("Text encoding {}".format(str(note)))
    logging.info("Text encoding")

    model = Roberta().to(device)

    if n_gpu > 1:
        new_state_dict = OrderedDict()
        for key, value in torch.load('../data/RoBERTa-encode/RoBERTa_{}_epoch_10_9.ckpt'.format(note)).items():
            name = key[7:]
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load('../data/RoBERTa-encode/RoBERTa_{}_epoch_10_9.ckpt'.format(note)))

    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    note_train = 'train_' + note
    note_test = 'test_' + note
    print('get train embedding')
    get_RoBERTa_embedding(model, trainDatasetPath, note=note_train)    # vuln_embedding_train, commit_embedding_train
    print('get test embedding')
    get_RoBERTa_embedding(model, testDatasetPath, note=note_test)