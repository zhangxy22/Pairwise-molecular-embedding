import csv
import os
import sys
import time
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
import shap
# from tensorflow.keras.models import load_model, Model
# from rdkit.Chem import AllChem
# from rdkit import Chem
from scipy.stats import pearsonr
import pandas as pd
from matplotlib import gridspec, ticker
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Embedding
# from tensorflow.keras import Sequential
# from tensorflow.keras.optimizers import Adam
# aesol regression model ========================
start_time = time.time()

# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''
trfile = open('数据记录.csv', 'r')
line = trfile.readline()
dataX = []
dataY = []
for line in trfile:
    line = line.rstrip().split(',')
    smiles = str(line[0])   #d的smiles
    #smiles = str(line[1])   #a的smiles
    mol = Chem.MolFromSmiles(smiles)
    print(smiles)
    print(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp = np.array(fp)
    print(fp)
    dataX.append(fp)
    val = float(line[2])  # pce
    dataY.append(val)
dataX = np.array(dataX)
print(dataX)
print(dataX.shape)
print(dataX[0])
np.save('fd_fp', dataX)
#np.save('fa_fp', dataX)
dataY = np.array(dataY)
np.save('fp_Y', dataY)
sys.exit()
'''
############################
dataX1 = np.load('fd_fp.npy')
dataX2 = np.load('fa_fp.npy')
dataX = np.hstack((dataX1,dataX2))
dataY = np.load('fp_Y.npy')
print(dataX.shape, dataY.shape)
FP_length = 1024
batch_size = 2
Max_len = 200
bit_size1 = 1024
bit_size = 2048
embedding_size = 100
# emb = tf.Variable(tf.random.uniform([bit_size1 + 1, embedding_size], -1, 1), dtype=tf.float32)
# pads = tf.constant([[1, 0], [0, 0]])
# embeddings = tf.pad(emb, pads)
data_x = []
data_y = []
for i in range(len(dataX)):
    fp = [0] * Max_len
    n_ones = 0
    for j in range(bit_size):
        if j < 1024:
            if dataX[i][j] == 1:
                fp[n_ones] = j + 1
                n_ones += 1
        else:
            if dataX[i][j] == 1:
                fp[n_ones] = j + 1 - 1024
                n_ones += 1
    data_x.append(fp)
    data_y.append([dataY[i]])
data_x = np.array(data_x, dtype=np.int32)
data_y = np.array(data_y, dtype=np.float32)
#
# value_info = defaultdict(lambda: {"count": 0, "coordinates": []})
#
# # 遍历二维数组，统计每个值的频率和坐标值
# for i, row in enumerate(data_x):
#     for j, value in enumerate(row):
#         value_info[value]["count"] += 1
#         value_info[value]["coordinates"].append((i, j))
#
# # 获取频率前十的值和对应的信息
# top_10_values_info = sorted(value_info.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
#
# # 打印结果
# for value, info in top_10_values_info:
#     count = info["count"]
#     coordinates = info["coordinates"][:2]  # 只获取前两个坐标
#     print(f"值 {value} 出现的频率是 {count}，坐标为 {coordinates}")
train_x_concat, test_x_concat, train_y_concat, test_y_concat, valid_x_concat, valid_y_concat = [], [], [], [], [], []
#计算两个分子相似性
def custom_distance(X1,X2,gamma_d,gamma_a):
    distance = 0.0
    # Calculate distances for FP
    ndesp1 = FP_length
    ndesp2 = FP_length + FP_length
    ndesp = FP_length + FP_length
    T_d = ( np.dot(np.transpose(X1[:ndesp1]),X2[:ndesp1]) ) / ( np.dot(np.transpose(X1[:ndesp1]),X1[:ndesp1]) + np.dot(np.transpose(X2[:ndesp1]),X2[:ndesp1]) - np.dot(np.transpose(X1[:ndesp1]),X2[:ndesp1]) )
    if np.allclose(np.dot(np.transpose(X1[ndesp1:ndesp2]), X1[ndesp1:ndesp2]), 0.0) or np.allclose(
            np.dot(np.transpose(X2[ndesp1:ndesp2]), X2[ndesp1:ndesp2]), 0.0):
        T_a = 0.0  # 在分母为零或接近零时将T_a设置为0
    else:
        T_a = (np.dot(np.transpose(X1[ndesp1:ndesp2]), X2[ndesp1:ndesp2])) / (
                    np.dot(np.transpose(X1[ndesp1:ndesp2]), X1[ndesp1:ndesp2]) + np.dot(np.transpose(X2[ndesp1:ndesp2]),
                                                                                        X2[ndesp1:ndesp2]) - np.dot(
                np.transpose(X1[ndesp1:ndesp2]), X2[ndesp1:ndesp2]))

    #T_a = ( np.dot(np.transpose(X1[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) ) / ( np.dot(np.transpose(X1[ndesp1:ndesp2]),X1[ndesp1:ndesp2]) + np.dot(np.transpose(X2[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) - np.dot(np.transpose(X1[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) )
    d_fp_d = 1 - T_d
    d_fp_a = 1 - T_a
    distance = distance + gamma_d*d_fp_d + gamma_a*d_fp_a
    return distance
def hspxy(x, y, test_size=0.2):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)
    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))
    Dc = np.zeros((M, M))
    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            xab = np.vstack((xa,xb))
            D[i, j] = custom_distance(xa,xb,1,1)
            Dy[i, j] = np.linalg.norm(ya - yb)
            Dc[i,j] = abs(1-pdist(xab,'cosine'))
    Dmax = np.max(D)
    Dymax = np.max(Dy)
    Dcmax = np.max(Dc)
    D = 1 + D / Dmax - Dc / Dcmax + Dy / Dymax
    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()
    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]
    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]
    m_complement = np.delete(np.arange(x.shape[0]), m)
    x_train = x[m, :]
    y_train = y_backup[m]
    x_test = x[m_complement, :]
    y_test = y_backup[m_complement]
    return x_train, x_test, y_train, y_test
train_x, test_x, train_y, test_y = hspxy(data_x, data_y, test_size=0.2)
#print(train_x.shape) 还是一个486*200的矩阵
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111,random_state=21)#21划分较好
train_size = len(train_x)
test_size = len(test_x)
valid_size = len(valid_x)
# ===================== model construction =======================
# Training data
# train_x, test_x, train_y, test_y, valid_x, valid_y = [], [], [], [], [], []
# Define the input shape
# input_shape = (Max_len, embedding_size, 1)
lr = 5e-4
# 定义模型参数
max_sequence_length = 200  # 输入文本的最大长度
max_words = 1024  # 词汇表的大小
filter_sizes = 5  # 卷积核的尺寸
num_filters = 64  # 每个尺寸的卷积核数量
embedding_size = 100
Max_len = 200
batch_size = 2


# 构建TextCNN模型
def My_model(x_train, y_train, x_test, y_test):
    main_input = tf.keras.Input(shape=(200,), dtype='float64')
    # 嵌入层（使用预训练的词向量）
    embedder = tf.keras.layers.Embedding(bit_size1 + 1, embedding_size, input_length=Max_len)
    embed = embedder(main_input)
    # 卷积层和池化层，设置卷积核大小 5
    cnn1 = tf.keras.layers.Conv1D(64, 5, padding='valid', strides=1, activation='relu')(embed)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=Max_len - 5 + 1)(cnn1)
    flat = tf.keras.layers.Flatten()(cnn)
    drop = tf.keras.layers.Dropout(0.5)(flat) #在池化层到全连接层之前可以加上dropout防止过拟合
    main_output = tf.keras.layers.Dense(1, activation='linear')(drop)
    model = tf.keras.Model(inputs=main_input, outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model
'''
# 创建模型
model = My_model(train_x, train_y, test_x, test_y)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), run_eagerly=True)
# Training
# 不需要将标签转换为独热编码，保持原始标签形式
labels = train_y
# one_hot_labels = keras.utils.to_categorical(train_y, num_classes=1)  # 将标签转换为one-hot编码
print(labels.shape, train_x.shape, train_y.shape)

model.fit(train_x, labels, batch_size=2, epochs=100, validation_data=(valid_x, valid_y), verbose=2)
# Saving the model
model.save("model.keras")
# 打印模型结构
model.summary()
'''
# Testing
# load the saved model
model = tf.keras.models.load_model("model.keras")
# Test the model
def r(a,b):
    r = np.sum((a-np.average(a))*(b-np.average(b)))/math.sqrt(np.sum((a-np.average(a))**2)*np.sum((b-np.average(b))**2))
    return r
def f(o,p,q):
    o_mean = np.average(o)
    a = q-o_mean
    ss = np.sum(a**2)
    b = p-q
    press = np.sum(b**2)
    f = 1 - press/ss
    return f
def plot_scatter(x_train, y_train, x_test, y_test):
    # general plot options
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    # 计算皮尔逊相关系数
    r, _ = pearsonr(x_test, y_test)
    rmse = np.sqrt(mean_squared_error(x_test, y_test))
    r2 =r2_score(x_test, y_test)
    Q2 = f(x_train, y_test, x_test)
    # rho, _ = spearmanr(x, y)
    ma = np.max([x_train.max(), x_test.max(), y_train.max(), y_test.max()]) + 1
    ax = plt.subplot(gs[0])
    ax.scatter(x_train, y_train, s=20, color='dimgrey', alpha=0.5)
    ax.scatter(x_test, y_test, s=20, color='r', alpha=0.3)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=10, length=5)
    ax.set_xlabel(u"PCE(Experimental)/%", size=18, labelpad=10)
    ax.set_ylabel(u"PCE(Predictive)/%", size=18, labelpad=10)
    ax.legend(['Training set', 'Test set'], fontsize=13, loc='upper left')
    ax.set_xlim(0, ma)
    ax.set_ylim(0, ma)
    ax.set_aspect('equal')
    ax.plot(np.arange(0, ma + 0.1, 0.1), np.arange(0, ma + 0.1, 0.1), color="gray", ls="--")
    ax.annotate(u'$r$ = %.2f' % r, xy=(0.15, 0.85), xytext=(0.7, 0.1), xycoords='axes fraction', size=13)
    # extra options in common for all plot types
    xtickmaj = ticker.MaxNLocator(5)
    xtickmin = ticker.AutoMinorLocator(5)
    ytickmaj = ticker.MaxNLocator(5)
    ytickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=22, pad=10, length=2)
    return
y_pred = model.predict(test_x)
y_train_pred =model.predict(train_x)
y_valid_pred = model.predict(valid_x)
print(train_y, y_train_pred)
print(1111)
print(valid_y, y_valid_pred)
print(test_y,y_pred)

# print("========== Result of Test Set ==========")
# print("RMSE = ", np.sqrt(mean_squared_error(test_y, y_pred)))
# print("R = ", r(test_y, y_pred))
# print("R2 = ", r2_score(test_y, y_pred))
# print("Q2 = ", f(train_y, y_pred, test_y))
# # Visualization
# y_train_pred = model.predict(train_x)
# test_y = test_y.ravel()
# y_pred = y_pred.ravel()
# plot_scatter(train_y, y_train_pred, test_y, y_pred)
# plt.subplots_adjust(bottom=0.2)
# plt.show()

'''

# 计算差异
# def find_top_min_differences(text_y, y_pred, num_samples=3):
#     differences = np.abs(text_y - y_pred)
#
#     # 找到最小的前 num_samples 个差异的索引
#     top_min_difference_indices = []
#     for _ in range(num_samples):
#         min_index = np.argmin(differences)
#         #min_index = np.argmax(differences)#最大差异性
#         top_min_difference_indices.append(min_index)
#         differences[min_index] = np.inf  # 将已经找到的差异置为正无穷大，以便找下一个最小差异
#         #differences[min_index] = -np.inf  # 将已经找到的差异置为负无穷大，以便找下一个最大差异
#
#     return top_min_difference_indices
def find_top_min_differences(text_y, y_pred, num_samples):
    # 计算差异
    differences = np.abs(text_y - y_pred)

    # 找到满足条件的最小的差异的索引，并将满足条件的索引加入列表
    top_min_difference_indices = []
    while len(top_min_difference_indices) < num_samples:
        min_index = np.argmin(differences)

        # 检查对应的 test_y 是否大于 10
        if test_y[min_index] > 10:
            top_min_difference_indices.append(min_index)

        # 将已经找到的差异置为正无穷大，以便找下一个最小差异
        differences[min_index] = np.inf

    return top_min_difference_indices

# 解释前 num_samples 个最小差异的样本
def explain_top_min_differences(text_x, text_y, y_pred, model, num_samples=5):
    # 找到前 num_samples 个最小差异的样本索引
    top_min_difference_indices = find_top_min_differences(text_y, y_pred, num_samples)
    # 创建一个 SHAP 解释器
    explainer = shap.Explainer(model, text_x)

    # 循环解释每个样本
    for i , sample_index in enumerate(top_min_difference_indices):
        print(f" 第{i + 1}小差异的索引: {sample_index}")
        print("真实值:", text_y[sample_index])
        print("预测值:", y_pred[sample_index])
        # 针对特定样本解释模型
        shap_values = explainer.shap_values(train_x[sample_index:sample_index + 1])
        shap_values_sample = shap_values[0, :]
        shap_values_positive = shap_values_sample  # 获取所有特征的 SHAP 值
        positive_indices = np.where(shap_values_positive > 0)[0]  # 找到所有正值 SHAP 值的索引
        positive_shap_values = shap_values_positive[shap_values_positive > 0]  # 所有正值的 SHAP 值

        # 找到样本中最大的 SHAP 值和对应的特征索引
        max_positive_shap_indices = np.argsort(positive_shap_values)[-5:][::-1]  # 找到前五个最大的 SHAP 值的索引
        max_positive_shap_values = positive_shap_values[max_positive_shap_indices]  # 前五个最大的 SHAP 值
        max_positive_feature_indices = positive_indices[max_positive_shap_indices]  # 前五个最大 SHAP 值对应的特征索引

        # 输出前五个最大的 SHAP 值和对应的特征索引
        for j in range(5):
            print(f'这是第{sample_index}样本的展示')
            feature_index = max_positive_feature_indices[j]
            shap_value = max_positive_shap_values[j]
            print(f"Top {j + 1} 特征索引 {feature_index}: SHAP 值 {shap_value}")

        print('这是序列号和对应的摩根指纹嵌入数字', max_positive_feature_indices,
              text_x[sample_index, max_positive_feature_indices])
        print("\n")


# 执行解释前五个最小差异的样本
explain_top_min_differences(test_x, test_y, y_pred, model, num_samples=5)



explainer = shap.Explainer(model, test_x)

#每个特征对所有测试样本的绘图
#shap.summary_plot(shap_values, test_x)

# 解释模型预测
sample_indices = list(range(len(test_x)))  # 所有测试样本
shap_values = explainer.shap_values(test_x[sample_indices])


# 计算每个样本的SHAP值的绝对值之和
shap_values_sum = np.sum(shap_values, axis=1)

# 找到影响最大的样本的索引
max_influence_sample_index = np.argmax(shap_values_sum)

# 输出最大影响的样本的索引和SHAP值统计信息
print("最大影响的样本索引:", max_influence_sample_index)
print("SHAP值统计信息:")
#print(shap_values[max_influence_sample_index])

# 可视化最大影响的样本
#shap.plots.waterfall(shap_values[max_influence_sample_index])
shap.summary_plot(shap_values[max_influence_sample_index], test_x[max_influence_sample_index])



# 计算所有测试样本的平均 SHAP 值
average_shap_values = np.mean(shap_values, axis=0)

# 找到对最终结果影响最大的特征的索引
max_influence_feature_index = np.argmax(average_shap_values)
# 获取最大的前二十个值的索引
top_20_indices = np.argpartition(average_shap_values, -20)[-20:]

# 打印对最终结果影响最大的特征
max_influence_feature = test_x[max_influence_feature_index]
print("对最终结果影响最大的特征索引是:", max_influence_feature_index)
print("对最终结果影响最大的特征:", max_influence_feature)

'''




