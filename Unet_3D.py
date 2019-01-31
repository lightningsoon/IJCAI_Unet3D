from keras import layers
from keras.utils import plot_model
from keras import models

import parameters


def get_unet3D(show=False):
    def conv_block(x, *ofs):
        for l in [
            layers.Conv3D(ofs[0], (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(ofs[1], (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ]:
            x = l(x)
        return x

    def down_module(x, o1, o2):
        x0 = conv_block(x, o1, o2)
        x1 = layers.MaxPooling3D((2, 2, 2), 2, padding='same')(x0)
        return x0, x1

    def up_module(x, o1, o2):
        x = conv_block(x, o1, o2)
        x = layers.Deconv3D(o2, (2, 2, 2), 2, padding='same')(x)
        return x

    o0 = 24
    inputs = layers.Input((None, parameters.img_rows, parameters.img_cols, 1))

    conv1, down_conv1 = down_module(inputs, o0, 2 * o0)
    conv2, down_conv2 = down_module(down_conv1, 2 * o0, 2 ** 2 * o0)
    conv3, down_conv3 = down_module(down_conv2, 2 ** 2 * o0, 2 ** 3 * o0)
    conv4, down_conv4 = down_module(down_conv3, 2 ** 3 * o0, 2 ** 4 * o0)
    up_conv4 = up_module(down_conv4, 2 ** 4 * o0, 2 ** 5 * o0)
    concat4 = layers.Concatenate()([conv4, up_conv4])
    up_conv3 = up_module(concat4, 2 ** 5 * o0, 2 ** 4 * o0)
    concat3 = layers.Concatenate()([conv3, up_conv3])
    up_conv2 = up_module(concat3, 2 ** 4 * o0, 2 ** 3 * o0)
    concat2 = layers.Concatenate()([conv2, up_conv2])
    up_conv1 = up_module(concat2, 2 ** 3 * o0, 2 ** 2 * o0)
    concat1 = layers.Concatenate()([conv1, up_conv1])
    y = conv_block(concat1, 2 ** 2 * o0, 2 * o0)
    y = layers.Conv3D(1, (1, 1, 1), padding='same')(y)
    y = layers.Activation('sigmoid')(y)
    model = models.Model(inputs=[inputs], outputs=[y])
    if show:
        model.summary()
        plot_model(model, parameters.plot_name, True)
        model.save(parameters.mo_name)
    return model


if __name__ == '__main__':
    get_unet3D(True)
'''
Total params: 64,486,561
Trainable params: 64,477,777
Non-trainable params: 8,784
'''
