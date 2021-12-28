# -*- coding: utf-8 -*-
# @Time    : 12/28/2021 2:04 PM
# @Author  : Isaac_Gao
# @File    : cifar10_data_reader.py
import _pickle as cPickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


test_dataset = unpickle("cifar10//test_batch")
test_label = test_dataset[b'labels']
test_datas = test_dataset[b'data']

dataset_info =unpickle("cifar10//batches.meta")
label_names = dataset_info [b'label_names']
pass

