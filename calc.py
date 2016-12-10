import nn
import numpy as np

def makeData(operators):
    inputdata = []
    outputdata = []
    for op in operators:
        for i in range(1, 10):
            for j in range(1, 10):
                if op == '+':
                    datumI = "{:02d}+{:02d}".format(i, j)
                    datumO = i + j
                elif op == '-':
                    datumI = "{:02d}-{:02d}".format(i, j)
                    datumO = i - j
                elif op == '*':
                    datumI = "{:02d}*{:02d}".format(i, j)
                    datumO = i * j
                elif op == '/':
                    datumI = "{:02d}/{:02d}".format(i, j)
                    datumO = i / j

                inputdata.append([int(datumI[0:2]), ord(datumI[2]), int(datumI[3:])])
                # inputdata.append([int(datumI[0]), int(datumI[2])])
                outputdata.append(datumO)

    else:
        inputdata = np.array(inputdata)
        outputdata = np.array(outputdata)
        outputdata = outputdata.reshape(outputdata.size, 1)
        # shuffle input, output data
        tmp = np.hstack((inputdata, outputdata))
        np.random.shuffle(tmp)
        inputdata, outputdata, tmp = np.hsplit(tmp, np.array((3, 4)))

    return(inputdata, outputdata)

if __name__ == '__main__':
    # NN = nn.NeuralNetwork(layer_num=(2, 3, 1))
    NN = nn.NeuralNetwork(layer_num=(3, 50, 1))

    operators = ('+', '-')
    inputdata, outputdata = makeData(operators=operators)
    # inputdata2, outputdata2 = makeData(operator='-')
    # inputdata = inputdata1 + inputdata2
    # outputdata = outputdata1 + outputdata2

    # "+", "-" 1e-3 1000, "*" 3e-5 10000, "/" 1e-4 10000
    NN.setparams(mu=3e-5, MaxEpoch=10000)
    NN.train(inputdata=inputdata, outputdata=outputdata)
    NN.save('weight.npz')
    NN.plot(type='global')

    # NN.load("weight.npz")

    ans = np.concatenate((inputdata.transpose(), NN.propagation(inputdata.transpose())), axis=0)
    ans = ans.transpose()
    print(ans)
