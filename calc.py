import nn
import numpy as np

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

if __name__ == '__main__':
    NN = nn.NeuralNetwork(layer_num=(2, 10, 2))

    operators = ('+', '-')
    inputdata, outputdata = makeData(operators=operators)

    NN.setparams(mu=1e-3, MaxEpoch=1000)
    NN.train(inputdata=inputdata, outputdata=outputdata)
    # NN.save('weight.npz')
    NN.plot(type='global')

    # NN.load("weight.npz")

    ans = np.concatenate((inputdata.transpose(), NN.propagation(inputdata.transpose())), axis=0)
    ans = ans.transpose()
    print(ans)
