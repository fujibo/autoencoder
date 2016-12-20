import nn_lbfgs as nn
import numpy as np
import scipy.io as spi
from PIL import Image

def dataVis(data):
    # 0 - 255であると推定
    scale = 255 / (np.max(data) - np.min(data))
    displacement = np.min(data)

    for i in range(data.shape[2]):
        datai = (data[:, :, i] - displacement) * scale
        print(datai)

        canvas = Image.new('L', (512, 512))
        for j in range(512):
            for k in range(512):
                canvas.putpixel((j, k), int(datai[j, k]))
        canvas.save('./DATA/image{:02d}.bmp'.format(i))


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
        outputdata = (outputdata + 10) / 30
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
        data = np.array(data)
        # Remove DC (mean of images).
        patches = data - np.mean(data, axis=0)

        # Truncate to +/-3 standard deviations and scale to -1 to 1
        pstd = 3 * np.std(data, ddof=1);
        patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd;

        # Rescale from [-1,1] to [0.1,0.9]
        patches = (patches + 1) * 0.4 + 0.1;
        print("make data")
        return patches

if __name__ == '__main__':
    NN = nn.NeuralNetwork(layer_num=(64, 25, 64))
    # NN = nn.NeuralNetwork(layer_num=(2, 9, 2))

    # print(loaddata().shape)
    # operators = ('+', '-')
    # inputdata, outputdata = makeData(operators=operators)

    data = loaddata()
    # print(data)
    # print(np.max(data))
    # print(np.min(data))

    NN.setparams(mu=3, MaxEpoch=400, TestRatio=0, lam=1e-4, beta=3, rho=0.01)
    # NN.train(inputdata=inputdata, outputdata=outputdata)
    NN.train(inputdata=data, outputdata=data)
    # NN.save('weight.npz')
    NN.plot(type='global')
    NN.visualize()
    # NN.load("weight.npz")
