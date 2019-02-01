from data import dataset
import SimpleITK as sitk
import numpy as np
from Unet_3D import get_unet3D
from parameters import we_name
from propress import preprocess


class Test(dataset):
    def __init__(self, path):
        super(Test, self).__init__(path)
        self.reader = sitk.ImageFileReader()
        self.metrics = []
        self.predict()
        with open('test_metrics.txt', 'w') as f:
            f.write(str(np.mean(self.metrics)))
            f.writelines([str(i) for i in self.metrics])

    def load_data(self, folders: list, shuffle: bool = True):
        # reader.GetRegisteredImageIOs()
        # if shuffle:
        #     # 训练集需要打乱数据
        #     random.seed(int(time.time()))
        #     random.shuffle(folders)
        for name in folders:
            self.reader.SetFileName(name[0])
            img_x = self.reader.Execute()
            x = sitk.GetArrayFromImage(img_x)
            self.reader.SetFileName(name[1])
            img_y = self.reader.Execute()
            y = sitk.GetArrayFromImage(img_y)
            yield (x, y)
        pass

    def predict(self):
        m = get_unet3D()
        m.load_weights(we_name)
        length = 32
        for x, y in self.load_data(self.file_list):
            x = preprocess(x)
            y = np.clip(y, 0, 1)
            x, y = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis]
            Y = np.zeros_like(x)
            for i in range(0, x.shape[0] - length, length):
                Y[i:i + length] = m.predict(np.array([x[i:i + length]]))[0]
            Y[x.shape[0] - length:] = m.predict(np.array([x[x.shape[0] - length:]]))[0]
            Y = np.round(Y).astype(np.uint8)
            self.calu_metric(y, Y)

    def calu_metric(self, y, Y):
        self.metrics.append(self.dice_metric(y, Y))

        pass

    def dice_metric(self, y_true, y_pred):
        smooth = 1
        y_true = np.clip(y_true, 0, 1).astype(np.uint8)
        y_pred = y_pred.astype(np.uint8)
        intersection = np.sum(y_true * y_pred, [1, 2, 3])
        unionsection = np.sum(y_true, [1, 2, 3]) + np.sum(y_pred, [1, 2, 3])
        return np.mean((2 * intersection + smooth) / (unionsection + smooth), axis=0)
        pass
