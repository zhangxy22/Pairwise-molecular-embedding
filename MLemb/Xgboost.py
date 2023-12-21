import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from matplotlib import ticker, gridspec
from pylab import*
from sklearn.metrics import mean_squared_error,r2_score
from deepforest import CascadeForestRegressor
trainpath = './featuremap-train.csv'
data1 = pd.read_csv(trainpath)
y_train = data1[['pce']]
x_train = data1.drop(['pce'],axis=1)

testpath = './featuremap-test.csv'
data2 = pd.read_csv(testpath)
y_test = data2[['pce']]
x_test = data2.drop(['pce'],axis=1)

xscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train)
x_test = xscaler.fit_transform(x_test)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

mpl.rcParams['font.sans-serif']=['SimHei']

def plot_scatter(x_train, y_train, x_test, y_test):
    # general plot options
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    r, _ = pearsonr(x_test, y_test)
    # rho, _ = spearmanr(x, y)
    ma = np.max([x_train.max(), x_test.max(), y_train.max(), y_test.max()]) + 1
    ax = plt.subplot(gs[0])
    ax.scatter(x_train, y_train, s=20, color='dimgrey', alpha=0.5)
    ax.scatter(x_test, y_test, s=20, color='r', alpha=0.3)
    ax.tick_params(axis='both', which='major', direction='in', labelsize=18, pad=10, length=5)
    ax.set_xlabel(u"PCE(Experimental)/%", size=18, labelpad=10)
    ax.set_ylabel(u"PCE(Predictive)/%", size=18, labelpad=10)
    ax.legend(['train', 'test'], fontsize=13, loc='upper left')
    # ax.set_xlabel(u"PCE / %", size=24, labelpad=10)
    # ax.set_ylabel(u'PCE$^{%s}$ / %s' %(SVR,"%"), size=24, labelpad=10)
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




# fit model no training data

model = XGBRegressor(max_depth=2,
                        learning_rate=0.1,
                        n_estimators=100,
                        objective='reg:linear',
                        booster='gbtree',
                        gamma=0,
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1,
                        reg_alpha=0,
                        reg_lambda=2,
                        random_state=0)


model.fit(x_train, y_train)
# make predictions for test data
y_pred = model.predict(x_test)
y_test=np.ravel(y_test)
y_pred=np.ravel(y_pred)
y_train_pred = model.predict(x_train)
plot_scatter(y_train,y_train_pred,y_test,y_pred)
plt.subplots_adjust(bottom=0.2)
#plt.title("XGBoost",size=18)
plt.show()

def r(a,b):
 r=np.sum((a-np.average(a))*(b-np.average(b)))/math.sqrt(np.sum((a-np.average(a))**2)*np.sum((b-np.average(b))**2))
 return r

import q2
Q2 = q2.f(y_train, y_pred, y_test)

print("----------------------Result of  model-----------------------")
print('RMSE=',np.sqrt(mean_squared_error(y_test,y_pred)))
print('r=',r(y_test,y_pred))
print('R2=',r2_score(y_test, y_pred))
print('Q2=',Q2)

def plot_learning_curves(model, x_train, y_train, x_test, y_test):

    train_errors, test_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_test_predict = model.predict(x_test)
        train_errors.append(np.sqrt(mean_squared_error(y_train_predict, y_train[:m])))
        test_errors.append(np.sqrt(mean_squared_error(y_test_predict, y_test)))
        if m == 400:
            train_r = np.sqrt(mean_squared_error(y_train_predict, y_train[:m]))
            test_r = np.sqrt(mean_squared_error(y_test_predict, y_test))
    plt.plot(np.sqrt(train_errors), "r", linewidth=1, label="train")
    plt.plot(np.sqrt(test_errors), "b", linewidth=1, label="test")
    plt.legend(loc="upper right", fontsize=14)
    #gap = test_r - train_r
    #plt.annotate(u'$gap$ = %.2f' % gap, xy=(0.15, 0.85), xytext=(0.79, 0.1),xycoords='axes fraction', size=13)

plt.xlabel("Training set size", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plot_learning_curves(model, x_train, y_train, x_test, y_test)
plt.show()




