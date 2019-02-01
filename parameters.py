import os
mo_name = 'model.h5'
we_name = 'weight.h5'
plot_name = 'Unet3d.pdf'
img_rows, img_cols = None, None
train_path = '../LITS/train'
validate_path = '../LITS/val'
logs_path = './logs'
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
