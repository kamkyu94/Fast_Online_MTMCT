import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import config as config
from nets.resnet_ibn_a import *
import utils.triplet as triplet
from data import TrainDataManager
import nets.estimator as estimator

# GPU setting
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Set Random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# Logging
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
log = open(config.log_path, 'w')

# DataManager and DataLoader for train and validation
# num_workers should be 1
tr_dm = TrainDataManager()
tr_dl = torch.utils.data.DataLoader(tr_dm, batch_size=config.batch_size, shuffle=False, num_workers=1, drop_last=True,
                                    prefetch_factor=500)

# Define model
model = resnet50_ibn_a() if config.model_name == 'resnet50_ibn_a' else resnet101_ibn_a()
model.load_param(config.pretrained)
model = estimator.Estimator(model, config.avg_type)
model = nn.DataParallel(model).cuda()

# Optimizer, Criterion
opt = torch.optim.Adam(params=model.parameters(), lr=config.init_lr, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss()
tri_loss = triplet.TripletLoss()

# Learning rate scheduler - cosine annealing
step_sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.milestones, gamma=0.1)


def train_one_epoch(epoch):
    # Prepare
    model.train()
    tr_ce_loss, tr_tri_loss, tr_ide_acc = 0, 0, 0
    tr_dm.initiate()
    total_iter_num = tr_dm.__len__() // config.batch_size

    # Start
    for idx, data in enumerate(tr_dl):
        # Set zero gradient
        opt.zero_grad()

        # Get train batch
        img, obj_id = data['img'].cuda(), data['label'].cuda()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=True):
            feat_tri, _, ide = model(img)

        # Loss, Back-propagation
        ce_loss_ = ce_loss(ide, obj_id)
        tri_loss_ = tri_loss(feat_tri.squeeze(), obj_id)
        loss = ce_loss_ + tri_loss_
        loss.backward()
        opt.step()

        # Loss, Accuracy
        tr_ce_loss += ce_loss_.item()
        tr_tri_loss += tri_loss_.item()
        prediction = ide.argmax(dim=1, keepdim=True)
        tr_ide_acc += prediction.eq(obj_id.view_as(prediction)).sum().item()

        # Logging
        if (idx + 1) % 100 == 0:
            print('Epoch: %d, Iter: %d / %d, CE Loss: %f, Tri Loss: %f'
                  % (epoch + 1, idx + 1, total_iter_num, tr_ce_loss / (idx + 1), tr_tri_loss / (idx + 1)))
            sys.stdout.flush()

    # Loss, Accuracy
    tr_ce_loss /= total_iter_num
    tr_tri_loss /= total_iter_num
    tr_ide_acc = tr_ide_acc / (total_iter_num * config.batch_size)

    return tr_ce_loss, tr_tri_loss, tr_ide_acc


def train():
    # Train
    for epoch in range(config.num_epoch):
        # Train
        tr_ce_loss, tr_tri_loss, tr_ide_acc = train_one_epoch(epoch)
        step_sch.step()

        # Logging
        print('Epoch %d, CE Loss: %f, Tri Loss: %f, IDE Acc: %f' % (epoch + 1, tr_ce_loss, tr_tri_loss, tr_ide_acc))
        log.write('Epoch %d, CE Loss: %f, Tri Loss: %f, IDE Acc: %f\n' % (epoch + 1, tr_ce_loss, tr_tri_loss, tr_ide_acc))
        log.flush()

        # Save weights
        if (epoch + 1) % 30 == 0:
            save_dict = {'opt': opt.state_dict(), 'model': model.state_dict()}
            torch.save(save_dict, config.save_path + config.model_name + '_' + config.avg_type
                       + '_' + str(int(epoch + 1)) + '.t7')

    # Close
    log.close()


if __name__ == "__main__":
    train()
