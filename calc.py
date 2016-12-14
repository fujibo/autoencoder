import nn
import numpy as np
import scipy.io as spi

def makeData(operators):
    inputdata = []
    outputdata = []
    for i in range(1, 10):
        for j in range(1, 10):
            datumI = [i, j]
            datumO = []
            for op in operators:
                if op == '+':
                    datumO.append(i + j)
                elif op == '-':
                    datumO.append(i - j)
                elif op == '*':
                    datumO.append(i * j)
                elif op == '/':
                    datumO.append(i / j)

                inputdata.append(datumI)
                outputdata.append(datumO)

    else:
        inputdata = np.array(inputdata)
        outputdata = np.array(outputdata)
        print(inputdata.shape)
        print(outputdata.shape)

        # shuffle input, output data
        tmp = np.hstack((inputdata, outputdata))
        np.random.shuffle(tmp)
        inputdata, outputdata, tmp = np.hsplit(tmp, np.array((2, 4)))

    return(inputdata, outputdata)

def loaddata():
    matdata = spi.loadmat("./DATA/IMAGES.mat")
    matdata = matdata['IMAGES']
    data = []
    for i in range(10000):
        img = np.random.randint(10)
        point = np.random.randint(0, 512-8, 2)
        data.append(matdata[point[0]:point[0]+8, point[1]:point[1]+8, img].flatten())
    else:
        print("make data")
        return np.array(data).transpose()

if __name__ == '__main__':
    NN = nn.NeuralNetwork(layer_num=(2, 10, 2))

    print(loaddata().shape)
    operators = ('+', '-')
    inputdata, outputdata = makeData(operators=operators)

    NN.setparams(mu=1e-3, MaxEpoch=1000, TestRatio=0)
    NN.train(inputdata=inputdata, outputdata=outputdata)
    # NN.save('weight.npz')
    NN.plot(type='global')

    # NN.load("weight.npz")

    ans = np.concatenate((inputdata.transpose(), NN.propagation(inputdata.transpose())), axis=0)
    ans = ans.transpose()
    print(ans)
