import torch
import torch.nn as nn
import torch.nn.functional as F


class ScharrAndLaplaceLoss(nn.Module):
    def __init__(self, device='cpu', use_mse=False):
        super(ScharrAndLaplaceLoss).__init__()
        self.scharr_x = torch.tensor([[-3, 0, -3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.use_mse = use_mse

    def forward(self, pred, target, mask=None):
        grad_pred_x = F.conv2d(pred, self.scharr_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.scharr_y, padding=1)

        grad_target_x = F.conv2d(target, self.scharr_x, padding=1)
        grad_target_y = F.conv2d(target, self.scharr_y, padding=1)

        laplace_pred = F.conv2d(pred, self.laplace_kernel, padding=1)
        laplace_target = F.conv2d(target, self.laplace_kernel, padding=1)

        if mask is None:
            loss_x = F.l1_loss(grad_pred_x, grad_target_x)
            loss_y = F.l1_loss(grad_pred_y, grad_target_y)
            if self.use_mse:
                loss_x += F.mse_loss(grad_pred_x, grad_target_x)
                loss_y += F.mse_loss(grad_pred_y, grad_target_y)
            loss_laplace = F.l1_loss(laplace_pred, laplace_target)
        else:
            loss_x = F.l1_loss(grad_pred_x[mask], grad_target_x[mask])
            loss_y = F.l1_loss(grad_pred_y[mask], grad_target_y[mask])
            if self.use_mse:
                loss_x += F.mse_loss(grad_pred_x[mask], grad_target_x[mask])
                loss_y += F.mse_loss(grad_pred_y[mask], grad_target_y[mask])
            loss_laplace = F.l1_loss(laplace_pred[mask], laplace_target[mask])

        loss = loss_x + loss_y + loss_laplace

        return loss
