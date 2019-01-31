from keras.optimizers import Adam
import os
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
import parameters
import Unet_3D
from tools import dice_coef_loss, dice_metric
import pynvml
from data import dataset
import json


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def lr_SCH(epochs, lr):
    if epochs <= 2:
        lr = 0.016
    elif epochs <= 4:
        lr = 0.002
    elif epochs != 0 and epochs % 4 == 0:
        # 每一新轮开始时,需要重置
        lr = 0.0016
        # print(lr,type(lr))
    else:
        lr = lr * 0.5
    return lr


class fitter():
    def __init__(self, gpu_nums, model_path, save_path, we_name):
        self.gpu_nums = gpu_nums
        self.model_path = model_path
        self.save_path = save_path
        self.we_name = we_name
        self.set_model()

    def set_model(self):
        # 模型
        self.single_m = None
        self.m = Unet_3D.get_unet3D()

        if self.gpu_nums >= 2:
            print('多核')
            self.single_m = self.m
            self.m = multi_gpu_model(self.m, num_gpus)
            pass
        if os.path.isfile(parameters.we_name):
            print('载入权重')
            self.m.load_weights(parameters.we_name)
        self.m.compile(optimizer=Adam(),
                       loss=dice_coef_loss, metrics=[dice_metric])

    def callback_func(self):
        # 画图
        tensor_bd = TensorBoard(batch_size=2 * num_gpus, write_graph=False, write_grads=False, write_images=False)
        # 模型保存
        if self.gpu_nums > 1:
            mcp = ParallelModelCheckpoint(self.single_m, self.save_path, period=1)
        else:
            mcp = ModelCheckpoint(self.save_path, period=1)
        # 学习率
        lrp = LearningRateScheduler(lr_SCH, 1)
        return [lrp, mcp, tensor_bd]
        pass

    def save_final(self):
        if self.gpu_nums > 1:
            self.single_m.save_weights(parameters.we_name)
        else:
            self.m.save_weights(parameters.we_name)


def train():
    train_data = dataset(parameters.train_path)
    # quit()
    val_data = dataset(parameters.validate_path)

    train_data.run_thread()
    val_data.run_thread()
    myfitter = fitter(num_gpus, model_path, save_path, parameters.we_name)
    h = myfitter.m.fit_generator(
        generator=train_data.generate_data(myfitter.gpu_nums * batch_size, train_times_each_data),
        steps_per_epoch=train_times_each_data // batch_size * train_data.files_number // myfitter.gpu_nums // epoch_scale_factor,
        epochs=epochs * epoch_scale_factor,
        callbacks=myfitter.callback_func(),
        validation_data=val_data.generate_data(myfitter.gpu_nums * batch_size, val_times_each_data),
        validation_steps=val_times_each_data // batch_size * val_data.files_number // myfitter.gpu_nums // epoch_scale_factor,
        initial_epoch=init_epoch)
    myfitter.save_final()
    hh = h.history
    with open('history.json', 'w') as f:
        json.dump(hh, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 设置GPU数量
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()  # 有几个卡用几个
    pynvml.nvmlShutdown()
    # 模型路径
    model_path = './model/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    init_epoch = len(os.listdir(model_path))
    save_path = model_path + 'weight-{epoch:03d}-{val_loss:.4f}.h5'
    #
    train_times_each_data = 135
    val_times_each_data = 30
    batch_size = 1
    epoch_scale_factor = 10
    epochs = 10
    train()
