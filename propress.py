import numpy as np
from skimage import measure


def preprocess(img):
    img = np.clip(img, -150, 250)
    min_nrrd_data = np.min(img)
    max_nrrd_data = np.max(img)
    img = 255 * (img - min_nrrd_data) / (max_nrrd_data - min_nrrd_data)
    return img


def max_connected_domain_3D(arr):
    # 取最大连通域
    labels = measure.label(arr)  # <1.2s
    t = np.bincount(labels.flatten())[1:]  # <1.5s
    max_pixel = np.argmax(t) + 1  # 位置变了,去除了0
    labels[labels != max_pixel] = 0
    labels[labels == max_pixel] = 1
    return labels.astype(np.uint8)


if __name__ == '__main__':
    import nrrd

    a, infor = nrrd.read('dataset/2/2_label_label_pre_25d_U.nrrd')
    a = max_connected_domain_3D(a)
    nrrd.write('dataset/2/2_label_label_pre_25d_U_reprocess5.nrrd', a, infor)
    print('ok')
