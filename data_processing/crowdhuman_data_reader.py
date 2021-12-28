# -*- coding: utf-8 -*-
# @Time    : 12/28/2021 2:45 PM
# @Author  : Isaac_Gao
# @File    : crowdhuman_data_reader.py

def crowdhuman(path):
    dataset=[]
    with open(path,'rb') as file:
        sample = file.readline()
        while sample:
            data = eval(sample)
            dataset.append(data)
            sample = file.readline()
    return dataset

dataset = crowdhuman("crowdhuman//annotation_val.odgt")
"""
ID:图片id
gtbox：
    tag ：分类标签
    hbox：head box
    vbox：可见身体box
    fbox：联想扩展身体box
"""
pass


