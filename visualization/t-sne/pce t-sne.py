from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

trainpath = './featuremap-all.csv'
data1 = pd.read_csv(trainpath)
y_train = data1[['pce']]
x_train = data1.drop(['pce'],axis=1)
y_train = np.ravel(y_train)

tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_train)
pca = PCA().fit_transform(x_train)
plt.scatter(pca[:, 0], pca[:, 1], c=y_train)
plt.xlabel(u"T-SNE1", size=15, labelpad=10)
plt.ylabel(u"T-SNE2", size=15, labelpad=10)
clb = plt.colorbar()
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 15,
        }
clb.set_label('PCE/%',fontdict=font,labelpad=10)
plt.show()