import nn_lbfgs
import numpy as np
import scipy.io as spi
from PIL import Image

def makeData():
    'make training data'
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
        spi.savemat("patches.mat", {"patches": patches.transpose()})
        return patches

if __name__ == '__main__':
    NN = nn_lbfgs.NeuralNetwork(layer_num=(64, 25, 64))

    data = makeData()
    # NN.load("weight.npz")

    NN.setparams(mu=3, MaxEpoch=10000, lam=1e-4, beta=3, rho=0.01)
    NN.train(inputdata=data, outputdata=data)

    NN.plot(type='global')
    NN.visualize()
    # NN.save('weight.npz')
