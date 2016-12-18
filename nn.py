'This module is for 3 layers newral network.'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spi
import numba
# import scipy.optimize as sp_opt
from PIL import Image

def activation_func(z, type="sigmoid"):
    '''activation function used in neural network
    default: sigmoid function
    '''
    if type == "sigmoid":
        return 1 / (1 + np.exp(-z))
    else:
        return z

def activation_difffunc(z, type="sigmoid"):
    'differential of activation function'
    if type == "sigmoid":
        return activation_func(z) * (1 - activation_func(z))
    else:
        return 1

class NeuralNetwork(object):
    """this is a class for 3 layers NeuralNetwork"""
    def __init__(self, layer_num, filename=None):
        if filename is None:
            'initialize network'
            self.INPUT_LAYER, self.HIDDEN_LAYER, self.OUTPUT_LAYER = layer_num
            # bias
            self.INPUT_LAYER += 1
            self.HIDDEN_LAYER += 1

            self.setparams()
            self.trainAccuracies = []
            self.testAccuracies = []
        else:
            f = open(filename, 'r')

    def initW(self):
        'initialize weight'
        const = (self.HIDDEN_LAYER-1) + (self.INPUT_LAYER-1) + 1
        tempW2 = np.random.rand(self.HIDDEN_LAYER-1, self.INPUT_LAYER-1) * 2 * np.sqrt(6/const) - np.sqrt(6/const)
        self.W2 = np.concatenate((tempW2, np.zeros((self.HIDDEN_LAYER-1, 1))), axis=1)

        const = (self.HIDDEN_LAYER-1) + self.OUTPUT_LAYER + 1
        self.W3 = np.concatenate((np.random.rand(self.OUTPUT_LAYER, self.HIDDEN_LAYER-1)  * 2 * np.sqrt(6/const) - np.sqrt(6/const), np.zeros((self.OUTPUT_LAYER, 1))), axis=1)
    @numba.jit
    def updateW(self, inputdatum, outputdatum):
        'update weight'
        gradW2, gradW3 = self.backpropagation(inputdatum, outputdatum)
        # gradW2, gradW3 = self.numericalGrad(inputdatum, outputdatum)
        self.W2 -= self.mu * gradW2
        self.W3 -= self.mu * gradW3

    def setparams(self, mu=1e-4, lam=1e-6, rho=0.05, beta=1e-6, MaxTrial=50, MaxEpoch=100, TestRatio=10):
        'set parameters'
        self.mu = mu
        self.lam = lam
        self.rho = rho
        self.beta = beta
        self.MaxTrial = MaxTrial
        self.MaxEpoch = MaxEpoch
        # percentage
        self.TestRatio = TestRatio

    def train(self, inputdata, outputdata):
        'train network'
        # check in row
        if inputdata.shape[0] != outputdata.shape[0]:
            print("input data size is NOT equal to output data size")
            exit(0)

        # replace m with self.inputDataNum
        # self.inputDataNum = inputdata.shape[0]

        datasize = inputdata.shape[0]
        testsize = int(datasize * self.TestRatio / 100)
        trainsize = datasize - testsize

        testdataI = inputdata[0:testsize]
        testdataO = outputdata[0:testsize]
        traindataI = inputdata[testsize:]
        traindataO = outputdata[testsize:]

        traindataI = traindataI.reshape(traindataI.shape[0], traindataI.shape[1], 1)
        traindataO = traindataO.reshape(traindataO.shape[0], traindataO.shape[1], 1)

        testdataI = testdataI.transpose()
        testdataO = testdataO.transpose()

        self.initW()

        for i in range(self.MaxEpoch):
            # self.checkgrad(traindataI[:,:,0].transpose(), traindataO[:,:,0].transpose())
            self.updateW(traindataI[:,:,0].transpose(), traindataO[:,:,0].transpose())
            # for j in range(self.MaxTrial):
            #     pickupiter = np.random.randint(trainsize)
            #     tdataI = traindataI[pickupiter]
            #     tdataO = traindataO[pickupiter]
            #     self.updateW(tdataI, tdataO)

            # varidation / 2
            self.trainAccuracies.append(self.cost(traindataI[:, :, 0].transpose(), traindataO[:, :, 0].transpose()) / (trainsize-1))
            print(i)
            print(self.trainAccuracies[-1])
            print(self.activeNum)
            # self.testAccuracies.append(self.cost(testdataI, testdataO) / (testsize-1))
    @numba.jit
    def cost(self, inData, outData):
        'cost function used in this NN'
        m = inData.shape[1]
        J =  0.5 / m * np.sum(np.square(outData - self.propagation(inData))) + self.lam * 0.5 * (np.sum(np.square(self.W2[:, 1:])) + np.sum(np.square(self.W3[:, 1:])))

        J += self.beta * np.sum(self.rho * np.log(self.rho/self.activeNum) + (1 - self.rho) * np.log((1-self.rho)/(1-self.activeNum)))
        return J

    def save(self, filename):
        'save network'
        np.savez(filename, w2=self.W2, w3=self.W3)

    def load(self, filename):
        npzfile = np.load(filename)
        print(npzfile.files)
        self.W2 = npzfile['w2']
        self.W3 = npzfile['w3']

    def plot(self, type='global'):
        'plot train accuracies and test accuracies'

        print(self.trainAccuracies[-1])
        # print(self.testAccuracies[-1])

        if type == 'global':
            plt.plot(range(self.MaxEpoch), self.trainAccuracies, label='train')
            # plt.plot(range(self.MaxEpoch), self.testAccuracies, label='test')
        else:
            plt.plot(range(10, self.MaxEpoch), self.trainAccuracies[10:], label='train')
            # plt.plot(range(10, self.MaxEpoch), self.testAccuracies[10:], label='test')

        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.savefig('figure.eps')
        # plt.show()

    def visualize(self):
        'visualize hidden unit feature'
        spi.savemat("w2.mat", {"w2": self.W2[:, 1:]})
        reg = np.sum(np.square(self.W2[:, 1:]), axis=1)
        reg = reg.reshape(reg.size, 1)
        visData = 1 / reg * self.W2[:, 1:]

        spi.savemat("visdata.mat", {"visdata": visData})
        k = 0
        for element in visData:
            length = int(np.sqrt(element.shape[0]))
            element = element.reshape(length, length)
            element = element * 255
            element = element.astype('uint8')
            img = Image.new("L", (length, length))
            for i in range(length):
                for j in range(length):
                    img.putpixel((i, j), element[i, j])
            else:
                img.save("bmp/vis{:02d}.bmp".format(k))
                k += 1

    @numba.jit
    def propagation(self, inputdata, type=None):
        'propagation in network'

        inputdata = np.concatenate((np.ones((1, inputdata.shape[1])), inputdata), axis=0)


        u2 = self.W2.dot(inputdata)

        x2 = activation_func(u2)
        x2 = np.concatenate((np.ones((1, x2.shape[1])), x2), axis=0)

        # rho^
        self.activeNum = np.mean(x2[1:,:], axis=1)

        u3 = self.W3.dot(x2)
        x3 = activation_func(u3)
        xs = (inputdata, x2, x3)
        us = (u2, u3)
        if type is None:
            return x3
        else:
            return (xs, us)
    @numba.jit
    def backpropagation(self, inputdatum, outputdatum):
        'propagation in network and return gradient'
        xs, us = self.propagation(inputdatum, 'forBP')

        x1, x2, x3 = xs
        u2, u3 = us


        m = x1.shape[1]

        gradEx3 = x3 - outputdatum
        gradW3 = 1/m * (gradEx3 * activation_difffunc(u3)).dot(x2.transpose())

        # regularization
        regW3 = np.hstack((np.zeros((gradW3.shape[0], 1)), self.W3[:, 1:]))
        gradW3 += self.lam * regW3

        # gradEx2 = (gradEx3 * activation_difffunc(u3)).transpose().dot(self.W3)
        # gradEx2 = gradEx2.reshape(gradEx2.size, 1)
        gradEx2 = self.W3.transpose().dot(gradEx3 * activation_difffunc(u3))
        # bias項のぶんだけ除く add term for sparse
        sparseTerm = self.beta * (-self.rho/self.activeNum + (1 - self.rho)/(1 - self.activeNum))
        sparseTerm = sparseTerm.reshape(sparseTerm.size, 1)

        gradW2 = 1/m * ((gradEx2[1:] + sparseTerm) * activation_difffunc(u2)).dot(x1.transpose())

        #regularization
        regW2 = np.hstack((np.zeros((gradW2.shape[0], 1)), self.W2[:, 1:]))
        gradW2 += self.lam * regW2


        return (gradW2, gradW3)

    def numericalGrad(self, inputdatum, outputdatum):
        '''get weights gradient by numerical way.
            this is only for checking, so please do not use.
        '''
        ngradW2 = np.zeros(self.W2.shape)
        ngradW3 = np.zeros(self.W3.shape)
        delta = 1e-6
        originW2 = self.W2.copy()
        originW3 = self.W3.copy()

        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                self.W2[i][j] += delta
                upper = self.cost(inputdatum, outputdatum)
                self.W2 = originW2.copy()
                self.W2[i][j] -= delta
                lower = self.cost(inputdatum, outputdatum)
                ngradW2[i][j] = (upper - lower) / (2 * delta)
                self.W2 = originW2.copy()

        for i in range(self.W3.shape[0]):
            for j in range(self.W3.shape[1]):
                self.W3[i][j] += delta
                upper = self.cost(inputdatum, outputdatum)
                self.W3 = originW3.copy()
                self.W3[i][j] -= delta
                lower = self.cost(inputdatum, outputdatum)
                ngradW3[i][j] = (upper - lower) / (2 * delta)
                self.W3 = originW3.copy()

        return (ngradW2, ngradW3)


    def checkgrad(self, inputdatum, outputdatum):
        'check gradient of weights'
        bgradW2, bgradW3 = self.backpropagation(inputdatum, outputdatum)
        print("backp end")
        ngradW2, ngradW3 = self.numericalGrad(inputdatum, outputdatum)
        print("numerical end")

        print(bgradW2.shape)
        print(bgradW2)
        print(ngradW2.shape)
        print(ngradW2)

        print(bgradW3.shape)
        print(bgradW3)
        print(ngradW3.shape)
        print(ngradW3)
        input()
