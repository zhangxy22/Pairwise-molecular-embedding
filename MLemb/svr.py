from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from matplotlib import ticker, gridspec
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics
from pylab import *
from scipy.spatial.distance import pdist
import math
from sklearn.svm import SVR

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

def msefunc(predictval, realval):
    print("RMSE = ", np.sqrt(metrics.mean_squared_error(realval, predictval)))
    #predictval = np.ravel(predictval)
    #realval = np.ravel(realval)
    #print("R = ",pearsonr(realval, predictval))
    return np.sqrt(metrics.mean_squared_error(realval, predictval))


def SVMResult(vardim, x, bound):
    X = x_train.tolist()
    y = y_train.tolist()
    c = x[0]
    e = x[1]
    g = x[2]
    clf = SVR(C=c, epsilon=e, gamma=g)
    clf.fit(X, y)
    predictval = clf.predict(x_test.tolist())
    return msefunc(predictval, y_test.tolist())


class GAIndividual:
    '''
    individual of genetic algorithm
    '''

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.

    def generate(self):
        '''
        generate a random chromsome for genetic algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                            (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = SVMResult(self.vardim, self.chrom, self.bound)


import random
import copy


class GeneticAlgorithm:
    '''
    The class for genetic algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 3))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        evolution process of genetic algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.maxfitness = np.max(self.fitness)

        self.trace[self.t, 0] = self.best.fitness
        self.trace[self.t, 1] = self.avefitness
        self.trace[self.t, 2] = self.maxfitness
        print("Generation %d: optimal function value is: %f; average function value is %f;max function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.trace[self.t, 2]))
        while (self.t < self.MAXGEN - 1):
            self.t += 1
            self.selectionOperation()
            self.crossoverOperation()
            self.mutationOperation()
            self.evaluate()
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.maxfitness = np.max(self.fitness)

            self.trace[self.t, 0] = self.best.fitness
            self.trace[self.t, 1] = self.avefitness
            self.trace[self.t, 2] = self.maxfitness
            print(
                "Generation %d: optimal function value is: %f; average function value is %f;max function value is %f" % (
                    self.t, self.trace[self.t, 0], self.trace[self.t, 1], self.trace[self.t, 2]))

        print("Optimal function value is: %f; " %
              self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))

        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()
            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[
                                             j] * self.params[2] + (1 - self.params[2]) * newpop[i + 1].chrom[j]
                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                                             (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                                                     mutatePos] - (
                                                             newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (
                                                             1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                                                     mutatePos] + (
                                                             self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (
                                                             1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop

    def printResult(self):
        '''
        plot the result of the genetic algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        y3 = self.trace[:, 2]
        # plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.plot(x, y3, 'b', label='max value')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        #plt.subplots_adjust(bottom=0.2)
        plt.xlabel("GENS")
        plt.ylabel("RMSE")
        plt.title("GA")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    bound = np.array([[0, 0, 0], [20, 2, 100]])
    ga = GeneticAlgorithm(10, 3, bound, 20, [0.9, 0.1, 0.5])
    ga.solve()

    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    start1 = time.time()
    model_svr = SVR( kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.34, epsilon=0.35, shrinking=True,
 cache_size=200, verbose=False, max_iter=- 1) #0.81 1.42
    model_svr.fit(x_train, y_train)
    y_pred = model_svr.predict(x_test)
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)
    end1 = time.time()

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.legend(['True', 'SVR'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.title("SVR")
    plt.show()

    mpl.rcParams['font.sans-serif'] = ['SimHei']


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
        # ax.annotate(u'$r$ = %.2f' % r, xy=(0.15, 0.85), xytext=(0.65, 0.2), xycoords='axes fraction', size=10)
        # ax.annotate(u'$RMSE$ = %.2f' % rmse, xy=(0.15, 0.85), xytext=(0.65, 0.15), xycoords='axes fraction', size=10)
        # ax.annotate(u'$R2$ = %.2f' % r2, xy=(0.15, 0.85), xytext=(0.65, 0.1), xycoords='axes fraction', size=10)
        # ax.annotate(u'$Q2$ = %.2f' % Q2, xy=(0.15, 0.85), xytext=(0.65, 0.05), xycoords='axes fraction', size=10)

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


    y_train_pred = model_svr.predict(x_train)
    plot_scatter(y_train, y_train_pred, y_test, y_pred)
    plt.subplots_adjust(bottom=0.2)
    #plt.title("SVR",size=20)
    plt.show()

    def r(a, b):
        r = np.sum((a - np.average(a)) * (b - np.average(b))) / math.sqrt(
            np.sum((a - np.average(a)) ** 2) * np.sum((b - np.average(b)) ** 2))
        return r


    import q2
    Q2 = q2.f(y_train, y_pred, y_test)

    print("----------------------Result of  model-----------------------")
    print('RMSE=', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('r=', r(y_test, y_pred))
    print('R2=', metrics.r2_score(y_test, y_pred))
    print('Q2=', Q2)



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
        # gap = test_r - train_r
        # plt.annotate(u'$gap$ = %.2f' % gap, xy=(0.15, 0.85), xytext=(0.79, 0.1),xycoords='axes fraction', size=13)


    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    # plt.title("HSPXY")
    plot_learning_curves(model_svr, x_train, y_train, x_test, y_test)
    plt.show()


