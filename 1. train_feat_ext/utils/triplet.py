import torch
from torch import nn
import torch.nn.functional as F


def pdist(v):
    dist = torch.norm(v[:, None] - v, p=2, dim=2)
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin='soft', sample=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sample = sample

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Pairwise distances
        dist = pdist(inputs)

        # Find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).byte().cuda()] = 0.

        if self.sample:
            # Weighted sample pos and negative to avoid outliers causing collapse
            posw = (dist + 1e-12) * mask_pos.float()
            posi = torch.multinomial(posw, 1)
            dist_p = dist.gather(0, posi.view(1, -1))

            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # This was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (dist + 1e-12)) * mask_neg.float()
            negi = torch.multinomial(negw, 1)
            dist_n = dist.gather(0, negi.view(1, -1))
        else:
            # Hard negative
            ninf = torch.ones_like(dist) * float('-inf')
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            nindex = torch.max(torch.where(mask_neg, -dist, ninf), dim=1)[1]
            dist_n = dist.gather(0, nindex.unsqueeze(0))

        # Calc loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == 'soft':
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.)
        loss = diff.mean()

        return loss
