# -*- coding: utf-8 -*-
# @Time    : 12/28/2021 12:42 PM
# @Author  : Isaac_Gao
# @File    : mnist_data_reader.py
import struct
import numpy as np


def labelReader(path):# 本地地址为"mnist//t10k-labels-idx1-ubyte"
    with open(path, 'rb') as labels_file:
        magic, n = struct.unpack('>II', labels_file.read(8))
        labels = np.fromfile(labels_file, dtype=np.uint8)
    return labels

def imageReader(path):
    with open(path, 'rb') as image_file:
        magic, num, rows, cols = struct.unpack('>IIII', image_file.read(16))
        images = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(-1, 784)
        return images

test_labels = labelReader("mnist//t10k-labels-idx1-ubyte")
test_images = imageReader("mnist//t10k-images-idx3-ubyte")
pass



