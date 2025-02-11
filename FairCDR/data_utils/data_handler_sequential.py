import pickle
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_sequential import SequentialDataset
import torch as t
import torch.utils.data as data
from os import path


class DataHandlerSequential:
    def __init__(self):
        if configs['data']['name'] == 1 and configs['target_data']['name'] == 2:
            predir = './datasets/sequential/1_2/1'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 1 and configs['target_data']['name'] ==3:
            predir = './datasets/sequential/1_3/1'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 3 and configs['target_data']['name'] ==1:
            predir = './datasets/sequential/3_1/3'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 3 and configs['target_data']['name'] ==2:
            predir = './datasets/sequential/3_2/3'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 2 and configs['target_data']['name'] ==1:
            predir = './datasets/sequential/2_1/2'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 2 and configs['target_data']['name'] ==3:
            predir = './datasets/sequential/2_3/2'
            configs['data']['dir'] = predir
        self.trn_file = path.join(predir, 'train.tsv')
        self.val_file = path.join(predir, 'test.tsv')
        self.tst_file = path.join(predir, 'test.tsv')
        self.max_item_id = 0
        if configs['data']['name'] == 1 and configs['target_data']['name'] == 2:
            predir = './datasets/sequential/1_2/2'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 1 and configs['target_data']['name'] ==3:
            predir = './datasets/sequential/1_3/3'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 3 and configs['target_data']['name'] ==1:
            predir = './datasets/sequential/3_1/1'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 3 and configs['target_data']['name'] ==2:
            predir = './datasets/sequential/3_2/2'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 2 and configs['target_data']['name'] ==1:
            predir = './datasets/sequential/2_1/1'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 2 and configs['target_data']['name'] ==3:
            predir = './datasets/sequential/2_3/3'
        self.target_trn_file = path.join(predir, 'train.tsv')
        self.target_val_file = path.join(predir, 'test.tsv')
        self.target_tst_file = path.join(predir, 'test.tsv')
        self.target_max_item_id = 0

    def _read_tsv_to_user_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": [], "gender": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            # skip header
            line = f.readline()
            while line:
                uid, seq, last_item,gender = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(item) for item in seq]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                user_seqs["gender"].append(int(gender))

                self.max_item_id = max(
                    self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs

    def _read_tsv_to_user_seqs_target(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": [], "gender": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            # skip header
            line = f.readline()
            while line:
                uid, seq, last_item,gender = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(item) for item in seq]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                user_seqs["gender"].append(int(gender))

                self.target_max_item_id = max(
                    self.target_max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs

    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        # item originally starts with 1
        configs['data']['item_num'] = self.max_item_id

    def _set_statistics_target(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['target_data']['user_num'] = user_num
        # item originally starts with 1
        configs['target_data']['item_num'] = self.target_max_item_id-self.max_item_id
        print(configs['target_data']['item_num'])

    def _seq_aug(self, user_seqs):
        user_seqs_aug = {"uid": [], "item_seq": [], "item_id": [], "gender": []}
        for uid, seq, last_item,gender in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"], user_seqs["gender"]):
            user_seqs_aug["uid"].append(uid)
            user_seqs_aug["item_seq"].append(seq)
            user_seqs_aug["item_id"].append(last_item)
            user_seqs_aug["gender"].append(gender)
            for i in range(1, len(seq)-1):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq[:i])
                user_seqs_aug["item_id"].append(seq[i])
                user_seqs_aug["gender"].append(gender)
        return user_seqs_aug

    def load_data(self):
        #source data
        user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
        user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
        self._set_statistics(user_seqs_train, user_seqs_test)
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            user_seqs_aug = self._seq_aug(user_seqs_train)
            trn_data = SequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug)
        else:
            trn_data = SequentialDataset(user_seqs_train)
        tst_data = SequentialDataset(user_seqs_test, mode='test')
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

        #target data
        user_seqs_train_target = self._read_tsv_to_user_seqs_target(self.target_trn_file)
        user_seqs_test_target = self._read_tsv_to_user_seqs_target(self.target_tst_file)
        self._set_statistics_target(user_seqs_train_target, user_seqs_test_target)
        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            user_seqs_aug_target = self._seq_aug(user_seqs_train_target)
            trn_data_target = SequentialDataset(user_seqs_train_target, user_seqs_aug=user_seqs_aug_target)
        else:
            trn_data_target = SequentialDataset(user_seqs_train_target)
        tst_data_target = SequentialDataset(user_seqs_test_target, mode='test')
        self.test_dataloader_target = data.DataLoader(
            tst_data_target, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader_target = data.DataLoader(
            trn_data_target, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

        