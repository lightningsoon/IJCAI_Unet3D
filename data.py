import SimpleITK as sitk
import glob
import pathlib
import random
import time
from itertools import cycle
import numpy as np
import queue
import threading
from propress import preprocess


def data_slice(xy: tuple):
    # ((n,512,512),(n,512,512))
    # 只取非0的层
    x, y = xy
    x, y = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis]
    y0 = np.where(np.sum(y, [1, 2]) > 0)
    up, down = np.max(y0), np.min(y0)
    print((up - down) // 10)
    for i in range(down, up + 1 - 16, 10):
        yield x[i:i + 16], y[i:i + 16]
    yield x[up + 1 - 16:up + 1], y[up + 1 - 16:up + 1]


class dataset():
    def __init__(self, path):
        # 读取目录，手动启动生产线程，手动获取迭代器
        self.path = path
        self.files_number = self.get_files()

    def get_files(self):
        '''
        文件夹名
        :return: list 文件路径xs,ys
        '''

        def sort(x):
            return int(list(filter(str.isdigit, x))[0])

        file_list_y = glob.glob(pathlib.Path.joinpath(self.path, 'segmatation-*.nii'))
        file_list_x = glob.glob(pathlib.Path.joinpath(self.path, 'volume-*.nii'))
        file_list_y = sorted(file_list_y, key=sort)
        file_list_x = sorted(file_list_x, key=sort)
        file_list = list(zip(file_list_x, file_list_y))
        self.file_list = file_list
        return len(file_list)
        pass

    def run_thread(self):
        self.q = queue.Queue(8)
        thread = threading.Thread(target=self.__load_data_thread, args=(self.file_list, self.q), daemon=True)
        thread.start()

    def __load_data_thread(self, folders: list, produce_queue: queue.Queue, shuffle: bool = True):
        reader = sitk.ImageSeriesReader()
        reader.SetImageIO('NiftiImageIO')
        if shuffle:
            # 训练集需要打乱数据
            random.seed(int(time.time()))
            random.shuffle(folders)
        for name in cycle(folders):
            img_x = reader.SetFileNames(name[0])
            x = sitk.GetArrayFromImage(img_x)
            img_y = reader.SetFileNames(name[1])
            y = sitk.GetArrayFromImage(img_y)
            produce_queue.put([x, y])

    def generate_data(self, batch_size=1, loop_num_each_data=20, length=16):
        # 读取并返回有值部分的数据

        while True:
            xy = self.q.get()
            datas = []
            for __ in range(loop_num_each_data // batch_size):
                x, y = xy
                x, y = preprocess(x), preprocess(y)
                x, y = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis]
                y0 = np.where(np.sum(y, [1, 2]) > 0)
                up, down = np.max(y0), np.min(y0)
                for _ in range(batch_size):
                    start = random.randint(down, up - length + 1)  # 可选择的上界
                    datas.append([x[start:start + length], y[start:start + length]])
                yield datas

        pass
