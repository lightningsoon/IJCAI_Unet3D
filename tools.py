from keras import backend as K

def dice_coef(y_true, y_pred):
    # Dice系数作为分割的成本函数
    # y_pred = K.round(y_pred) #没有梯度
    smooth = 1
    intersection = K.sum(y_true * y_pred, [1, 2, 3])
    unionsection = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    return K.mean((2 * intersection + smooth) / (unionsection + smooth), axis=0)


def dice_metric(y_true, y_pred):
    y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
    y_pred = K.round(y_pred)
    smooth = 0.0001
    intersection = K.sum(y_true * y_pred, [1, 2, 3])
    unionsection = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    return K.mean((intersection) / (unionsection - intersection + smooth), axis=0)
    pass


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)