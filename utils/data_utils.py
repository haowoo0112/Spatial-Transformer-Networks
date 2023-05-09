import numpy as np
import torch
from torch.utils import data
import sys
sys.path.append("..")
import config as cfg
import logging


# logging.getLogger().setLevel(logging.INFO)

def get_data_loader():
    '''
    :return: train_loader, test_loader
    '''
    mnist_cluttered = np.load("./data/mnist_sequence1_sample_5distortions5x5.npz")
    X_train = torch.from_numpy(mnist_cluttered['X_train'])
    y_train = torch.from_numpy(mnist_cluttered['y_train'])
    X_valid = torch.from_numpy(mnist_cluttered['X_valid'])
    y_valid = torch.from_numpy(mnist_cluttered['y_valid'])
    X_test = torch.from_numpy(mnist_cluttered['X_test'])
    y_test = torch.from_numpy(mnist_cluttered['y_test'])

    X_train, y_train = to_torch_formart(X_train, y_train, data_type='Training')
    X_test = torch.cat([X_test, X_valid], dim=0)
    y_test = torch.cat([y_test, y_valid], dim=0)
    X_test, y_test = to_torch_formart(X_test, y_test, data_type='Testing')
    
    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=cfg.test_batch_size)
    # for i, train_data in enumerate(train_loader):
    #     x_data, label = train_data
    #     print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))
    return train_loader, test_loader

def to_torch_formart(img, target, data_type=''):
    '''
    :param img: (b, l)
    :param target: (b, 1)
    :param data_type: string
    :return: (b, c, h, w), (b)
    '''
    batch_size = img.size(0)
    img = img.view(batch_size, cfg.channel, cfg.height, cfg.width)
    target = target.view(batch_size)
    print(data_type, img.size(), target.size())
    return img, target
