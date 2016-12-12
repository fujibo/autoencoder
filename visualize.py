import scipy.io as spi
import numpy as np
from PIL import Image

if __name__ == '__main__':
    data = spi.loadmat("./DATA/IMAGES_RAW.mat")["IMAGESr"]
    print(data.shape)
    # 0 - 255であると推定
    scale = 255 / (np.max(data) - np.min(data))
    displacement = np.min(data)

    for i in range(data.shape[2]):
        datai = (data[:, :, i] - displacement) * scale
        print(datai)

        canvas = Image.new('L', (512, 512))
        for j in range(512):
            for k in range(512):
                canvas.putpixel((j, k), datai[j, k])
        canvas.save('./DATA/image{:02d}.bmp'.format(i))
