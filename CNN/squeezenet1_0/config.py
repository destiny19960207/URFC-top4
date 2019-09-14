#coding=utf-8
import warnings

class DefaultConfigs(object):
    env='default'

    model_name = "0626_debug"

    train_data = "D:/final/train_image/image/"  # where is your train images data
    test_data = "D:/final/test_image/"  # your test data
    train_vis = "D:/final/npy/0_9/npy"  # where is your train visits data
    test_vis = "D:/final/npy/test_visit"
    load_model_path = None
    weights = "checkpoints/"
    best_models = "checkpoints/best_models/"
    train_csv = "preliminary/train.csv"
    test_csv = "preliminary/test.csv"

    start_fold=1
    start_epoch=0

    num_kf = 5
    k_fold = True
    mix_up = False
    num_classes = 9

    lr = 0.00004
    batch_size = 256
    epochs = 20   #根据训练集准确率调整
    step =20
    alpha = 0.5
    
def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
