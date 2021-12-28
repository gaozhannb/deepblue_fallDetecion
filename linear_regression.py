# -*- coding: utf-8 -*-
# @Time    : 12/26/2021 2:26 PM
# @Author  : Isaac_Gao
# @File    : linear_regression.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class l1_regularization():
    def __init__(self, alpha):
        self.alpha = alpha

    # l1正则部分求和
    def __call__(self, w):
        loss = np.sum(np.fabs(w))
        return self.alpha * loss

    # l1的梯度
    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():

    def __init__(self, alpha):
        self.alpha = alpha

    # l2正则部分求和
    def __call__(self, w):
        loss = w.T.dot(w)
        return self.alpha * 0.5 * float(loss)

    # l2的梯度
    def grad(self, w):
        return self.alpha * w


class LinearRegression():
    """
    Parameters:
    -----------
    n_iterations: int
        梯度下降的轮数
    learning_rate: float
        梯度下降学习率
    regularization: l1_regularization or l2_regularization or None
        正则化
    gradient: Bool
        是否采用梯度下降法或正规方程法。
        若使用了正则化，暂只支持梯度下降
    """

    def __init__(self, n_iterations=3000, learning_rate=0.00005, regularization=None, gradient=True):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        if regularization == None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    # X为 （m x n）, w 为（n x 1）， y为（m x 1）在权重初始化的时候把b=0加到了w里边所以变成(（n+1） x 1)
    #  在数据x生成的时候相应的加了值为1的一列到X中变成（m x (n+1)）,最后的y不变。
    def initialize_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)  # （目标向量，插入位置，插入的数值，插入的维度）
        pass

    def fit(self, X, y):
        m_sample, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_sample, 1))
        self.training_error = []
        if self.gradient == True:
            # 梯度下降求w
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                # 计算loss
                loss = np.mean(0.5 * (y - y_pred) ** 2) + self.regularization(self.w)
                self.training_error.append(loss)
                # X.T.dot（y_pred - y)，计算梯度
                w_grad = X.T.dot(y_pred - y) + self.regularization.grad(self.w)
                self.w = self.w - w_grad * self.learning_rate
        else:
            # 正常解方程求出的w
            X = np.matrix(X)
            y = np.matrix(y)
            XTX = X.T.dot(X)
            XTXIXT = XTX.I.dot(XT)
            self.w = XTXIXT.dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == '__main__':
    # x为样本特征，y为样本输出，共100个样本，每个样本一个特征
    X, y = make_regression(n_samples=100, n_features=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # 可自行设置模型参数，如正则化，梯度下降轮数学习率等。不设置默认的数值
    model = LinearRegression(regularization=l2_regularization(alpha=0.5))
    model.fit(X_train, y_train)

    # 画图显示
    n = len(model.training_error)
    plt.plot(range(n), model.training_error, label='Training Error')
    plt.title('Error Plot')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    mse = mean_squared_error(y_test, y_pred)
    print('>>>>>>>>>>>>>>mse:{}>>>>>>>>>>>>>'.format(mse))

    y_pred_line = model.predict(X)

    train = plt.scatter(X_train, y_train)
    test = plt.scatter(X_test, y_test)
    plt.plot(X, y_pred_line, color='g', lw=2, label='Prediction')
    plt.title('LinearRegression')
    plt.xlabel('Day')
    plt.ylabel('Temp')
    plt.show()
