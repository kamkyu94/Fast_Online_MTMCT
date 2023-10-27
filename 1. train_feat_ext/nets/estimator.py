import config
import torch.nn as nn
import torch.nn.functional as F


class Estimator(nn.Module):
    def __init__(self, backbone, avg_type):
        super(Estimator, self).__init__()
        self.act = nn.ReLU(inplace=True)

        # Construct backbone network
        self.backbone = backbone

        # Global Average Pooling, Dropout
        self.avg = nn.AdaptiveAvgPool2d(1) if avg_type == 'gap' else self.gem
        self.drop = nn.Dropout(0.5)
   
        # BNNeck
        self.bnn = nn.BatchNorm2d(2048)
        nn.init.constant_(self.bnn.weight, 1)
        nn.init.constant_(self.bnn.bias, 0)
        self.bnn.bias.requires_grad_(False)
        
        # IDE
        self.fc_ide = nn.Linear(2048, config.num_ide_class)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, patch):
        # Extract appearance feature
        feat_tri = self.avg(self.backbone(patch))

        # BNNeck
        feat_infer = self.bnn(self.drop(feat_tri))

        # IDE
        feat_ide = feat_infer.view(feat_infer.size(0), -1)
        ide = self.fc_ide(feat_ide)

        return feat_tri, feat_infer, ide
