import git
import os
import logging

from tqdm import tqdm

from util import *

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AutoModel

from transformers import logging as lg
lg.set_verbosity_error()

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')     # added by me


class TextDataset(Dataset):
    def __init__(self, df):
        self.labels = torch.tensor(df['label'])
        df['desc_id'] = df['desc_id'].apply(eval)
        self.input1 = list(df['desc_id'].apply(torch.tensor))
        self.input1 = pad_sequence(self.input1).T.to(torch.int64)
        df['mess_id'] = df['mess_id'].apply(eval)
        self.input2 = list(df['mess_id'].apply(torch.tensor))
        self.input2 = pad_sequence(self.input2).T.to(torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input1 = self.input1[idx]
        input2 = self.input2[idx]
        sample = (input1, input2, label)
        return sample


class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = AutoModel.from_pretrained('../pretrained-model/BERT')
        self.linear = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(64, 2)

    def forward(self, input1, input2):
        out1 = self.linear2(self.linear(self.bert(input1)[1]))
        out2 = self.linear2(self.linear(self.bert(input2)[1]))
        out = torch.cat((out1, out2), 1)
        out = self.linear3(out)
        return out


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


def dataProcess(df, tokenizer, columns=['mess', 'desc']):
    for col in columns:
        df[col + '_token'] = df[col].apply(tokenizer.tokenize)  # use bert tokenizer
        df[col + '_id'] = df[col + '_token'].apply(
            tokenizer.convert_tokens_to_ids)                    # convert id to id of bert vocabulary table
        df[col + '_id'] = df[col + '_id'].apply(lambda x: x[:128])
        df.drop([col], axis=1, inplace=True)                    # delete col
    return df


def prepare_encoding():
    if not os.path.exists('../data/BERT-encode/TextEmbedding.csv'):
        df = pd.read_csv('../data/Dataset_150.csv')
        df = df[['cve', 'repo', 'commit']]
        df = add_desc(df)
        df = add_mess(df)
        tokenizer = BertTokenizer.from_pretrained('../pretrained-model/BERT')    # load bert tokenizer
        df = dataProcess(df, tokenizer)     # use bert tokenizer
        df.to_csv('../data/BERT-encode/TextEmbedding.csv', index=False)


def create_encode_dataset(train_df,
                          test_df,
                          trainDatasetPath='../data/BERT-encode/temp_enc_train',
                          testDatasetPath='../data/BERT-encode/temp_enc_test'):
    prepare_encoding()
    df = pd.read_csv('../data/BERT-encode/TextEmbedding.csv')
    train_df = pd.merge(left=train_df,
                        right=df,
                        on=['cve', 'repo', 'commit'])
    test_df = pd.merge(left=test_df,
                       right=df,
                       on=['cve', 'repo', 'commit'])

    trainDataset = TextDataset(train_df)
    testDataset = TextDataset(test_df)
    savefile(trainDataset, trainDatasetPath)
    savefile(testDataset, testDatasetPath)


def train_enc(trainDatasetPath,
              testDatasetPath=None,
              criterion=None,
              optimizer=None,
              num_epochs=20,
              batch_size=20,
              shuffle=False,
              num_workers=1,
              learning_rate=2e-5, note='idx-1'):
    trainDataset = readfile(trainDatasetPath)
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle)

    model = TextModel().to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        t1 = time.time()
        model.train()
        loss_sum = 0
        bar = tqdm(trainDataLoader, total=len(trainDataLoader))
        for i, (data1, data2, label) in enumerate(bar):
            data1 = data1.to(device)
            data2 = data2.to(device)
            label = label.to(device)
            pred = model(data1, data2)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()
        t2 = time.time()
        logging.info('Epoch [{:2}/{:2}], Loss: {:.4f}, Time: {:4}s'.format(
            epoch + 1, num_epochs, loss.item(), int(t2 - t1)))
        evaluation(model, trainDataLoader)
        logging.info("")
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(),
                       '../data/BERT-encode/model_{}_epoch_{}_{}.ckpt'.format(note, num_epochs, epoch))


def evaluation(model, dataloader):
    model.eval()
    TP, FP, FN, TN, cnt = 0, 0, 0, 0, 0
    with torch.no_grad():
        for i, (data1, data2, label) in enumerate(dataloader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            label = label.to(device)
            label_size = data1.size()[0]
            pred = model(data1, data2)
            pred_label = pred.argmax(axis=1)
            for item1, item2 in zip(pred_label, label):
                item1 = int(item1)
                item2 = int(item2)
                if item1 == item2 and item1 == 1:
                    TP += 1
                elif item1 == item2:
                    TN += 1
                elif item1 == 1:
                    FP += 1
                else:
                    FN += 1
            cnt += label_size

    TN = cnt - TP - FP - FN
    # precision = TP / (TP + FP)
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / cnt
    logging.info("precision:{:.4f}, recall: {:.4f}, accuracy: {:.4f}".format(
        precision, recall, accuracy))
    logging.info("TP = {:4d}, FP = {:4d}, FN = {:4d}, TN = {:4d}".format(
        TP, FP, FN, TN))


def get_embedding(model,
                  dataset_path,
                  batch_size=20,
                  shuffle=False,
                  num_workers=1,
                  note='data', 
                  outpath='../data/BERT-encode/'):
    dataset = readfile(dataset_path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    model.eval()
    data1_embedding = []
    data2_embedding = []
    result = []
    with torch.no_grad():
        for i, (data1, data2, label) in enumerate(dataloader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            out1 = model.linear2(model.linear(model.bert(data1)[1]))
            out2 = model.linear2(model.linear(model.bert(data2)[1]))
            data1_embedding.extend(out1.cpu().numpy())
            data2_embedding.extend(out2.cpu().numpy())
    data1_embedding = np.array(data1_embedding)
    data2_embedding = np.array(data2_embedding)
    savefile(data1_embedding, outpath + 'vuln_embedding_' + note)
    savefile(data2_embedding, outpath + 'commit_embedding_' + note)


def BERT_encoding(train_df, test_df, note):
    if not os.path.exists('../data/BERT-encode'):
        os.makedirs('../data/BERT-encode/')

    trainDatasetPath = '../data/BERT-encode/enc_train_' + note
    testDatasetPath = '../data/BERT-encode/enc_test_' + note
    print("Create dataset {}".format(str(note)))
    logging.info("Create dataset")
    create_encode_dataset(train_df, test_df, trainDatasetPath, testDatasetPath)     # trainDataset, testDataset

    print("Train encoding module {}".format(str(note)))
    logging.info("Train encoding module")
    train_enc(trainDatasetPath=trainDatasetPath, note=note)    # bert model fine-tune

    print("Text encoding {}".format(str(note)))
    logging.info("Text encoding")
    model = TextModel().to(device)
    model.load_state_dict(torch.load('../data/BERT-encode/model_{}_epoch_20_19.ckpt'.format(note)))  # load model

    note_train = 'train_' + note
    note_test = 'test_' + note
    get_embedding(model, trainDatasetPath, note=note_train)    # vuln_embedding_train, commit_embedding_train
    get_embedding(model, testDatasetPath, note=note_test)      # vuln_embedding_test, commit_embedding_test
