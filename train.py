import torch
from torch import nn
from torch.autograd import Variable
import torchvision.utils as vutils
from utils.data_utils import get_data_loader
from models.model import Network
import config as cfg
import logging
import os


def calc_acc(x, y):
    x = torch.max(x, dim=-1)[1]
    accuracy = sum(x == y) / x.size(0)
    return accuracy

logging.getLogger().setLevel(logging.INFO)
logging.info('Mode: %s' % cfg.mode)
if not os.path.exists(cfg.model_dir):
    os.mkdir(cfg.model_dir)
if not os.path.exists(cfg.transform_img_dir):
    os.mkdir(cfg.transform_img_dir)

train_loader, test_loader = get_data_loader()
train_batch_nb = len(train_loader)
test_batch_nb = len(test_loader)

print('Train batch_nb:%d' % train_batch_nb)
print('Test batch_nb:%d' % test_batch_nb)

net = Network(mode=cfg.mode)
if torch.cuda.is_available():
    net.cuda(cfg.cuda_num)

opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)
loss_func = nn.CrossEntropyLoss()

for epoch_idx in range(cfg.epoch):
    # ========================== Training Model =============================
    net.train()
    for i, train_data in enumerate(train_loader):
        train_img, train_target = train_data
        train_img = Variable(train_img)
        train_target = Variable(train_target)

        # if torch.cuda.is_available():
        #     train_img = train_img.cuda(cfg.cuda_num)
        #     train_target = train_target.cuda(cfg.cuda_num)

        _, predict = net(train_img)

        loss = loss_func(predict, train_target)
        net.zero_grad()
        loss.backward()
        opt.step()

        acc = calc_acc(predict.cpu().data, train_target.cpu().data)

        # if batch_idx % cfg.show_train_result_every_batch == 0:
        #     logging.info('epoch[%d/%d] batch[%d/%d] loss:%.4f acc:%.4f'
        #                  % (epoch_idx, cfg.epoch, batch_idx, train_batch_nb, loss.data[0], acc))

    # ========================== Testing Model =============================
    if (epoch_idx + 1) % cfg.test_every_epoch == 0:
        net.eval()
        total_loss = 0
        total_acc = 0
        for i, test_data in enumerate(test_loader):
            test_img, test_target = test_data
            batch_size = test_img.size(0)

            test_img = Variable(test_img, volatile=True)
            test_target = Variable(test_target, volatile=True)

            if torch.cuda.is_available():
                test_img = test_img.cuda(cfg.cuda_num)
                test_target = test_target.cuda(cfg.cuda_num)

            transform_img, predict = net(test_img)

            loss = loss_func(predict, test_target)
            acc = calc_acc(predict.cpu().data, test_target.cpu().data)

            total_loss += loss
            total_acc += acc

            if cfg.mode == 'stn':
                img_list = []
                for idx in range(batch_size):
                    img_list.append(test_img[idx])
                    img_list.append(transform_img[idx])
                output_img = torch.stack(img_list)

                vutils.save_image(output_img.data, os.path.join(cfg.transform_img_dir, '%d.png' % batch_idx),
                                  nrow=20)

        mean_loss = total_loss / test_batch_nb
        mean_acc = total_acc / test_batch_nb
        logging.info('========= Testing: epoch[%d/%d] loss:%.4f acc:%.4f' % (epoch_idx, cfg.epoch, mean_loss.data[0], mean_acc))

    if (epoch_idx + 1) % cfg.save_model_every_epoch == 0:
        state_dict = net.state_dict()
        torch.save(state_dict, os.path.join(cfg.model_dir, cfg.model_name % cfg.mode))