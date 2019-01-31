import SimpleITK as sitk
import glob
import os
import random
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
            # print(os.path.basename(x),''.join(list(filter(str.isdigit, os.path.basename(x)))))
            # print(int(''.join(list(filter(str.isdigit, os.path.basename(x))))))
            return int(''.join(list(filter(str.isdigit, os.path.basename(x)))))

        file_list_y = glob.glob(os.path.join(self.path, 'segmentation-*.nii'))
        file_list_x = glob.glob(os.path.join(self.path, 'volume-*.nii'))
        file_list_y = sorted(file_list_y, key=sort)
        file_list_x = sorted(file_list_x, key=sort)
        file_list = list(zip(file_list_x, file_list_y))
        self.file_list = file_list
        # print(file_list)
        return len(file_list)
        pass

    def run_thread(self):
        self.q = queue.Queue(10)
        thread = threading.Thread(target=self.__load_data_thread, args=(self.file_list, self.q), daemon=True)
        thread.start()

    def __load_data_thread(self, folders: list, produce_queue: queue.Queue, shuffle: bool = True):
        reader = sitk.ImageFileReader()
        # reader.GetRegisteredImageIOs()
        # if shuffle:
        #     # 训练集需要打乱数据
        #     random.seed(int(time.time()))
        #     random.shuffle(folders)
        for name in cycle(folders):
            reader.SetFileName(name[0])
            img_x = reader.Execute()
            x = sitk.GetArrayFromImage(img_x)
            reader.SetFileName(name[1])
            img_y = reader.Execute()
            y = sitk.GetArrayFromImage(img_y)
            # print(y.shape)
            produce_queue.put([x, y])

    def generate_data(self, batch_size=1, loop_num_each_data=135, length=160):
        # 读取并返回有值部分的数据
        z_len = 64
        while True:
            xy = self.q.get()
            x, y = xy
            x = preprocess(x).astype(np.float32)
            y = np.clip(y, 0, 1).astype(np.float32)  # 标签有0，1，2
            # 选择适适的z高度
            y0 = np.where(np.sum(y, (1, 2)) > 0)
            up, down = np.max(y0), np.min(y0)
            height = up - down
            if height < z_len:
                z_len = height - height % 16  # 取离16倍数最近的整数
                if z_len == 0:
                    continue
            else:
                z_len = 64
            x, y = x[:, :, :, np.newaxis], y[:, :, :, np.newaxis]

            # 对每个数据循环
            for __ in range(loop_num_each_data // batch_size):
                datas = [[], []]
                for _ in range(batch_size):
                    start = random.randint(down, up - z_len + 1)
                    acme = random.randint(0, 512 - length)
                    # print(start,stop)
                    datas[0].append(x[start:start + z_len, acme:acme + length, acme:acme + length])
                    datas[1].append(y[start:start + z_len, acme:acme + length, acme:acme + length])
                # print(np.array(datas).shape)
                yield np.array(datas)

        pass
