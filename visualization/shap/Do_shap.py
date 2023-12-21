import time
import numpy as np
from scipy.spatial.distance import pdist
import shap
from sklearn.model_selection import train_test_split
import tensorflow as tf
start_time = time.time()
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
train_x_concat, test_x_concat, train_y_concat, test_y_concat, valid_x_concat, valid_y_concat = [], [], [], [], [], []

def custom_distance(X1,X2,gamma_d,gamma_a):
    distance = 0.0
    # Calculate distances for FP
    ndesp1 = FP_length
    ndesp2 = FP_length + FP_length
    ndesp = FP_length + FP_length
    T_d = ( np.dot(np.transpose(X1[:ndesp1]),X2[:ndesp1]) ) / ( np.dot(np.transpose(X1[:ndesp1]),X1[:ndesp1]) + np.dot(np.transpose(X2[:ndesp1]),X2[:ndesp1]) - np.dot(np.transpose(X1[:ndesp1]),X2[:ndesp1]) )
    if np.allclose(np.dot(np.transpose(X1[ndesp1:ndesp2]), X1[ndesp1:ndesp2]), 0.0) or np.allclose(
            np.dot(np.transpose(X2[ndesp1:ndesp2]), X2[ndesp1:ndesp2]), 0.0):
        T_a = 0.0
    else:
        T_a = (np.dot(np.transpose(X1[ndesp1:ndesp2]), X2[ndesp1:ndesp2])) / (
                    np.dot(np.transpose(X1[ndesp1:ndesp2]), X1[ndesp1:ndesp2]) + np.dot(np.transpose(X2[ndesp1:ndesp2]),
                                                                                        X2[ndesp1:ndesp2]) - np.dot(
                np.transpose(X1[ndesp1:ndesp2]), X2[ndesp1:ndesp2]))
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
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111,random_state=21)
train_size = len(train_x)
test_size = len(test_x)
valid_size = len(valid_x)
# ===================== model construction =======================
# Training data
# train_x, test_x, train_y, test_y, valid_x, valid_y = [], [], [], [], [], []
# Define the input shape
# input_shape = (Max_len, embedding_size, 1)
lr = 5e-4
max_sequence_length = 200
max_words = 1024
filter_sizes = 5
num_filters = 64
embedding_size = 100
Max_len = 200
batch_size = 2
def My_model(x_train, y_train, x_test, y_test):
    main_input = tf.keras.Input(shape=(200,), dtype='float64')
    embedder = tf.keras.layers.Embedding(bit_size1 + 1, embedding_size, input_length=Max_len)
    embed = embedder(main_input)
    cnn1 = tf.keras.layers.Conv1D(64, 5, padding='valid', strides=1, activation='relu')(embed)
    cnn = tf.keras.layers.MaxPooling1D(pool_size=Max_len - 5 + 1)(cnn1)
    flat = tf.keras.layers.Flatten()(cnn)
    drop = tf.keras.layers.Dropout(0.5)(flat)
    main_output = tf.keras.layers.Dense(1, activation='linear')(drop)
    model = tf.keras.Model(inputs=main_input, outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model
'''
model = My_model(train_x, train_y, test_x, test_y)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), run_eagerly=True)
# Training
labels = train_y
# one_hot_labels = keras.utils.to_categorical(train_y, num_classes=1) 
print(labels.shape, train_x.shape, train_y.shape)
model.fit(train_x, labels, batch_size=2, epochs=100, validation_data=(valid_x, valid_y), verbose=2)
# Saving the model
model.save("model.keras")
model.summary()
'''
# Testing
# load the saved model
model = tf.keras.models.load_model("model.keras")
# Test the model

y_pred = model.predict(test_x)
y_train_pred =model.predict(train_x)
y_valid_pred = model.predict(valid_x)


def find_top_min_differences(text_y, y_pred, num_samples):
    differences = np.abs(text_y - y_pred)
    top_min_difference_indices = []
    while len(top_min_difference_indices) < num_samples:
        min_index = np.argmin(differences)
        if test_y[min_index] > 10:
            top_min_difference_indices.append(min_index)
        differences[min_index] = np.inf
    return top_min_difference_indices
def explain_top_min_differences(text_x, text_y, y_pred, model, num_samples=5):
    top_min_difference_indices = find_top_min_differences(text_y, y_pred, num_samples)
    explainer = shap.Explainer(model, text_x)
    for i , sample_index in enumerate(top_min_difference_indices):
        print(f" {i + 1}Index of small differences: {sample_index}")
        print("True value:", text_y[sample_index])
        print("Predicted value:", y_pred[sample_index])
        shap_values = explainer.shap_values(train_x[sample_index:sample_index + 1])
        shap_values_sample = shap_values[0, :]
        shap_values_positive = shap_values_sample
        positive_indices = np.where(shap_values_positive > 0)[0]
        positive_shap_values = shap_values_positive[shap_values_positive > 0]
        max_positive_shap_indices = np.argsort(positive_shap_values)[-5:][::-1]
        max_positive_shap_values = positive_shap_values[max_positive_shap_indices]
        max_positive_feature_indices = positive_indices[max_positive_shap_indices]
        for j in range(5):
            print(f'This is{sample_index}presentation of samples')
            feature_index = max_positive_feature_indices[j]
            shap_value = max_positive_shap_values[j]
            print(f"Top {j + 1} aspect indexing {feature_index}: SHAP Value {shap_value}")
        print('This is the serial number and the corresponding Morgan fingerprint embedding number:', max_positive_feature_indices,
              text_x[sample_index, max_positive_feature_indices])
        print("\n")
explain_top_min_differences(test_x, test_y, y_pred, model, num_samples=5)









