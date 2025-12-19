import torch
import torch.nn as nn
import torch.nn.functional as F


class SWTLoss(nn.Module):
    """
    A Theory for Multiresolution Signal Decomposition: The Wavelet Representation
    https://dl.acm.org/doi/10.1109/34.192463
    """
    def __init__(self, device='cpu', use_mse=False):
        super(SWTLoss).__init__()
        self.use_mse = use_mse

        s2 = 1.0 / torch.sqrt(torch.tensor(2.0)).cuda()
        h = torch.tensor([s2, s2], dtype=torch.float32).to(device)
        g = torch.tensor([s2, -s2], dtype=torch.float32).to(device)

        LL_kernel = torch.outer(h, h)
        LH_kernel = torch.outer(h, g)
        HL_kernel = torch.outer(g, h)
        HH_kernel = torch.outer(g, g)
        self.kernels = torch.stack([LL_kernel, LH_kernel, HL_kernel, HH_kernel], dim=0).unsqueeze(1)

    def forward(self, pred, target, mask=None):
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='reflect')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='reflect')

        grad_pred = F.conv2d(pred_pad, self.kernels)
        target_pred = F.conv2d(target_pad, self.kernels)

        if mask is None:
            loss = F.l1_loss(grad_pred, target_pred)
            if self.use_mse:
                loss += F.mse_loss(grad_pred, target_pred)
        else:
            loss = F.l1_loss(grad_pred[mask], target_pred[mask])
            if self.use_mse:
                loss += F.mse_loss(grad_pred[mask], target_pred[mask])

        return loss
