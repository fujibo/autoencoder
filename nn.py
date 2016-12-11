'This module is for 3 layers newral network.'
import numpy as np
import matplotlib.pyplot as plt

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
        self.W2 = np.random.rand(self.HIDDEN_LAYER, self.INPUT_LAYER)
        self.W3 = np.random.rand(self.OUTPUT_LAYER, self.HIDDEN_LAYER)

    def updateW(self, inputdatum, outputdatum):
        'update weight'
        gradW2, gradW3 = self.backpropagation(inputdatum, outputdatum)
        # gradW2, gradW3 = self.numericalGrad(inputdatum, outputdatum)
        self.W2 -= self.mu * gradW2
        self.W3 -= self.mu * gradW3

    def setparams(self, mu=1e-4, MaxTrial=50, MaxEpoch=100, TestRatio=10):
        'set parameters'
        self.mu = mu
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

        datasize = inputdata.shape[0]
        testsize = int(datasize * self.TestRatio / 100)
        trainsize = datasize - testsize

        testdataI = inputdata[0:testsize]
        testdataO = outputdata[0:testsize]
        traindataI = inputdata[testsize:]
        traindataO = outputdata[testsize:]

        testdataI = testdataI.transpose()
        testdataO = testdataO.transpose()

        self.initW()

        for i in range(self.MaxEpoch):
            for j in range(self.MaxTrial):
                pickupiter = np.random.randint(trainsize)
                # self.checkgrad(traindataI[pickupiter], traindataO[pickupiter])
                self.updateW(traindataI[pickupiter], traindataO[pickupiter])

            self.trainAccuracies.append(np.sum(np.square(traindataO.transpose() - self.propagation(traindataI.transpose()))) / (trainsize-1))

            self.testAccuracies.append(np.sum(np.square(testdataO - self.propagation(testdataI)))/ (testsize-1))

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
        print(self.testAccuracies[-1])

        if type == 'global':
            plt.plot(range(self.MaxEpoch), self.trainAccuracies, label='train')
            plt.plot(range(self.MaxEpoch), self.testAccuracies, label='test')
        else:
            plt.plot(range(10, self.MaxEpoch), self.trainAccuracies[10:], label='train')
            plt.plot(range(10, self.MaxEpoch), self.testAccuracies[10:], label='test')

        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.savefig('figure.eps')
        plt.show()

    def propagation(self, inputdata, type=None):
        'propagation in network'
        if len(inputdata.shape) == 1:
            inputdata = inputdata.reshape(inputdata.size, 1)
        inputdata = np.concatenate((np.ones((1, inputdata.shape[1])), inputdata), axis=0)

        u2 = self.W2.dot(inputdata)
        # u2_1 = W2[0].dot(inputdata)
        # u2_2 = W2[1].dot(inputdata)
        x2 = activation_func(u2)
        u3 = self.W3.dot(x2)
        x3 = activation_func(u3, "id")
        xs = (inputdata, x2, x3)
        us = (u2, u3)
        if type is None:
            return x3
        else:
            return (xs, us)

    def backpropagation(self, inputdatum, outputdatum):
        'propagation in network and return gradient'
        xs, us = self.propagation(inputdatum, 'forBP')

        x1, x2, x3 = xs
        u2, u3 = us
        x1 = x1.reshape(x1.size, 1)
        x2 = x2.reshape(x2.size, 1)
        x3 = x3.reshape(x3.size, 1)
        u2 = u2.reshape(u2.size, 1)
        u3 = u3.reshape(u3.size, 1)
        outputdatum = outputdatum.reshape(outputdatum.size, 1)
        gradEx3 = x3 - outputdatum
        # print(gradEx3.shape)
        # print(u3.shape)
        # print(x2.shape)
        gradW3 = (gradEx3 * activation_difffunc(u3, "id")).dot(x2.transpose())

        gradEx2 = (gradEx3 * activation_difffunc(u3, "id")).transpose().dot(self.W3)
        gradEx2 = gradEx2.reshape(gradEx2.size, 1)
        gradW2 = (gradEx2 * activation_difffunc(u2)).dot(x1.transpose())

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
                upper = 0.5 * (outputdatum - self.propagation(inputdatum)) ** 2
                self.W2 = originW2.copy()
                self.W2[i][j] -= delta
                lower = 0.5 * (outputdatum - self.propagation(inputdatum)) ** 2
                ngradW2[i][j] = (upper - lower) / (2 * delta)
                self.W2 = originW2.copy()

        for i in range(self.W3.shape[0]):
            for j in range(self.W3.shape[1]):
                self.W3[i][j] += delta
                upper = 0.5 * (outputdatum - self.propagation(inputdatum)) ** 2
                self.W3 = originW3.copy()
                self.W3[i][j] -= delta
                lower = 0.5 * (outputdatum - self.propagation(inputdatum)) ** 2
                ngradW3[i][j] = (upper - lower) / (2 * delta)
                self.W3 = originW3.copy()

        return (ngradW2, ngradW3)


    def checkgrad(self, inputdatum, outputdatum):
        'check gradient of weights'
        bgradW2, bgradW3 = self.backpropagation(inputdatum, outputdatum)
        ngradW2, ngradW3 = self.numericalGrad(inputdatum, outputdatum)

        print(bgradW2.shape)
        print(bgradW2)
        print(ngradW2.shape)
        print(ngradW2)

        print(bgradW3.shape)
        print(bgradW3)
        print(ngradW3.shape)
        print(ngradW3)
        input()
