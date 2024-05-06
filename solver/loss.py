# ////////////////////////////////////////////////////////////////////////////
# // This file is part of CRefNet. For more information
# // see <https://github.com/JundanLuo/CRefNet>.
# // If you use this code, please cite our paper as
# // listed on the above website.
# //
# // Licensed under the Apache License, Version 2.0 (the “License”);
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an “AS IS” BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# ////////////////////////////////////////////////////////////////////////////


from abc import abstractmethod
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.sparse
import numpy as np
import torch.nn.functional as F

from solver.metrics_intrinsic_images import compute_DSSIM

from utils import image_util


class MaskedLoss(nn.Module):
    def __init__(self, mode="L1"):
        super(MaskedLoss, self).__init__()
        if mode == "L1":
            self.criteria = nn.L1Loss(reduction='sum')
        elif mode == "L2":
            self.criteria = nn.MSELoss(reduction="sum")
        else:
            raise Exception(f"Not support mode: {mode}")

    def forward(self, pred, target, valid_mask):
        if not pred.shape==target.shape==valid_mask.shape:
            raise Exception(f"Inconsistent dimensions: {pred.shape}; {target.shape}; {valid_mask.shape}")
        loss = self.criteria(pred.mul(valid_mask), target.mul(valid_mask)) / torch.sum(valid_mask).clamp(min=1e-6)
        return loss


class LaplaceFilter_5D(nn.Module):
    def __init__(self):
        super(LaplaceFilter_5D, self).__init__()
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        edge = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ])
        edge_k = edge
        edge_k = torch.from_numpy(edge_k).to(torch.float32).view(1, 1, 5, 5)
        self.edge_conv.weight = nn.Parameter(edge_k, requires_grad=False)

        if True:
            self.mask_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
            mask_k = np.array([
                [0, 0, 0.077, 0, 0],
                [0, 0.077, 0.077, 0.077, 0],
                [0.077, 0.077, 0.077, 0.077, 0.077],
                [0, 0.077, 0.077, 0.077, 0],
                [0, 0, 0.077, 0, 0]
            ])
            mask_k = torch.from_numpy(mask_k).to(torch.float32).view(1, 1, 5, 5)
            self.mask_conv.weight = nn.Parameter(mask_k, requires_grad=False)

        for param in self.parameters():
            param.requires_grad = False

    def apply_laplace_filter(self, x, mask=None):
        out = self.edge_conv(x)
        if mask is not None:
            out_mask = self.mask_conv(mask)
            out_mask[out_mask < 0.95] = 0
            out_mask[out_mask >= 0.95] = 1
            out = torch.mul(out, out_mask)
        else:
            out_mask = None
        return out, out_mask

    def forward(self, x, mask=None):
        out, out_mask = self.apply_laplace_filter(x[:, 0:1, :, :], mask[:, 0:1, :, :] if mask is not None else None)
        for idx in range(1, x.size(1)):
            d_out, d_out_mask = self.apply_laplace_filter(x[:, idx:idx+1, :, :],
                                                          mask[:, idx:idx+1, :, :] if mask is not None else None)
            out = torch.cat((out, d_out), 1)
            if d_out_mask is not None:
                out_mask = torch.cat((out_mask, d_out_mask), 1)

        return out, out_mask


class L1ImageGradientLoss(nn.Module):
    def __init__(self, step=2):
        super(L1ImageGradientLoss, self).__init__()
        self.step = step

    def forward(self, pred, target, mask):
        step = self.step

        N = torch.sum(mask)
        diff = pred - target
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:, :, 0:-step, :] - diff[:, :, step:, :])
        v_mask = torch.mul(mask[:, :, 0:-step, :], mask[:, :, step:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, :, :, 0:-step] - diff[:, :, :, step:])
        h_mask = torch.mul(mask[:, :, :, 0:-step], mask[:, :, :, step:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / 2.0
        gradient_loss = gradient_loss / (N + 1e-6)

        return gradient_loss


class SecondOrderGradLoss(nn.Module):
    def __init__(self):
        super(SecondOrderGradLoss, self).__init__()
        self.laplace = LaplaceFilter_5D()

    def forward(self, pred, target, mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions in SecondOrderGradLoss"
        lap_pred, mask_lap = self.laplace(pred, mask)
        lap_target, _ = self.laplace(target, mask)
        diff = (lap_pred - lap_target) * mask_lap
        tot_loss = torch.sum(torch.abs(diff)) / torch.sum(mask_lap + 1e-6)
        return tot_loss


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, order=1, stride=2, step=4, mode="L1", eq_threshold=0.0):
        super(MultiScaleGradientLoss, self).__init__()
        if order == 1:
            # self.gradient_filter = K.filters.SpatialGradient(mode="sobel", order=1, normalized=True)
            self.gradient_filter = self.neighbor_gradient
        else:
            raise Exception(f"Not support order {order}")
        if mode == "L1":
            self.criteria = nn.L1Loss(reduction='sum')
        elif mode == "L2":
            self.criteria = nn.MSELoss(reduction="sum")
        else:
            raise Exception(f"Not supports mode {mode}.")
        self.stride = stride
        self.step = step
        self.suppress_gradient = eq_threshold > 1e-5
        self.eq_threshold = eq_threshold

    @staticmethod
    def neighbor_gradient(img, mask, suppress_equal=False, eq_threshold=0.0):
        step = 1
        v_gradient = img[:, :, 0:-step, :] - img[:, :, step:, :]
        if suppress_equal:
            # base = torch.min(img[:, :, 0:-step, :], img[:, :, step:, :])
            v_gradient = v_gradient * (v_gradient.abs() > eq_threshold).to(torch.float32)
        v_mask = mask[:, :, 0:-step, :] * mask[:, :, step:, :]
        v_gradient = v_gradient * v_mask

        h_gradient = img[:, :, :, 0:-step] - img[:, :, :, step:]
        if suppress_equal:
            # base = torch.min(img[:, :, :, 0:-step], img[:, :, :, step:])
            h_gradient = h_gradient * (h_gradient.abs() > eq_threshold).to(torch.float32)
        h_mask = mask[:, :, :, 0:-step] * mask[:, :, :, step:]
        h_gradient = h_gradient * h_mask
        return (v_gradient, h_gradient), (v_mask, h_mask)

    # def weight_adjustment(self, target_grad, mask):
    #     mask_eq = (target_grad.abs() < 0.95).to(torch.float32) * mask
    #     mask_ineq = (1.0 - mask_eq) * mask
    #     sum_eq, sum_ineq = mask_eq.sum(), mask_ineq.sum()
    #     mask_eq *= 1.5 / (sum_eq / sum_ineq)
    #     weight = mask_eq + mask_ineq
    #     return weight

    def forward(self, pred, target, mask):
        if not pred.shape == target.shape == mask.shape:
            raise Exception(f"Inconsistent dimensions: {pred.shape}; {target.shape}; {mask.shape}")
        pred = pred.mul(mask)
        target = target.mul(mask)
        loss = torch.tensor(0.0, dtype=torch.float32, device=pred.device)
        for i in range(0, self.step):
            pred_grads, grad_masks = self.gradient_filter(pred, mask, False)
            target_grads, _ = self.gradient_filter(target, mask, self.suppress_gradient, self.eq_threshold)

            loss += self.criteria(pred_grads[0], target_grads[0]) / grad_masks[0].sum().clamp(min=1e-6) * 0.5
            loss += self.criteria(pred_grads[1], target_grads[1]) / grad_masks[1].sum().clamp(min=1e-6) * 0.5

            pred = pred[:, :, ::self.stride, ::self.stride]
            target = target[:, :, ::self.stride, ::self.stride]
            mask = mask[:, :, ::self.stride, ::self.stride]

        return loss / self.step


class PixelWiseLoss(nn.Module):
    def __init__(self, w_2):
        super(PixelWiseLoss, self).__init__()
        self.w_2 = w_2

    def forward(self, pred, target, mask):
        if not pred.shape == target.shape == mask.shape:
            raise Exception(f"Inconsistent dimensions: {pred.shape}; {target.shape}; {mask.shape}")
        n_mask = mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)
        diff = (pred - target).mul(mask)
        loss = (diff ** 2).sum(dim=[1, 2, 3]) / n_mask
        if abs(self.w_2) > 1e-6:
            ss_diff = diff.sum(dim=[1, 2, 3]) ** 2
            loss_2 = ss_diff / (n_mask ** 2).clamp(min=1e-6)
            loss = loss - self.w_2 * loss_2
        return loss.mean()


def DenseIntrinsicCriterion(color_rep, key, suppress_c, suppress_i,
                            w_c=1.0, w_i=1.0,
                            w_value=1.0, w_grad=40.0, w_dssim=0.0):
    if color_rep == "rgI":
        return DenseIntrinsicCriterion_rgI(key, suppress_c, suppress_i,
                                           w_c, w_i, w_value, w_grad, w_dssim)
    elif color_rep == "RGB":
        return DenseIntrinsicCriterion_RGB(key, suppress_i,
                                           w_value, w_grad, w_dssim)
    else:
        raise Exception("Unknown color representation: {}".format(color_rep))


class DenseIntrinsicCriterion_rgI(nn.Module):
    Criteria = namedtuple("Criteria", ["rg", "w_rg", "I", "w_I", "w"])
    key = None  # should be "R" or "S"
    w_value = 1.0
    w_grad = 40.0
    w_dssim = 0.0
    w_c = 1.0
    w_i = 1.0

    def __init__(self, key, suppress_c, suppress_i,
                 w_c=1.0, w_i=1.0,
                 w_value=1.0, w_grad=40.0, w_dssim=0.0):
        '''
            :param suppress_c: suppression threshold for the chromaticity
            :param suppress_i: suppression threshold for the intensity
        '''
        super(DenseIntrinsicCriterion_rgI, self).__init__()
        self.key = key
        self.w_value, self.w_grad, self.w_dssim = w_value, w_grad, w_dssim
        self.w_c, self.w_i = w_c, w_i
        # p = 1.0 if self.scale_invrc_mode == "direct_intrinsics" else 0
        p = 0
        self.value_criterion = self.Criteria(
            PixelWiseLoss(0), self.w_c,
            PixelWiseLoss(p), self.w_i,
            self.w_value
        )
        self.grad_criterion = self.Criteria(
            MultiScaleGradientLoss(order=1, stride=2, step=4, mode="L1", eq_threshold=suppress_c), self.w_c,
            MultiScaleGradientLoss(order=1, stride=2, step=4, mode="L1", eq_threshold=suppress_i), self.w_i,
            self.w_grad
        )
        print(f"Build DenseIntrinsicCriterion_rgI:\n"
              f"\tsuppression=({suppress_c}, {suppress_i}), \n"
              f"\tweights rg, I: ({self.w_c}, {self.w_i}), \n"
              f"\tweights: ({self.w_value}, {self.w_grad}, {self.w_dssim})")

    def generate_compared_pair(self, color_rep, pred, gt_RGB, mask):
        # prediction
        pred_rgI = pred[f"direct_linear_out_{self.key}"]
        # compared_pred_R_log = pred[f"direct_log_out_{self.key}"]

        # ground truth
        assert gt_RGB.shape[2:] == mask.shape[2:] == pred_rgI.shape[2:]
        # gt_RGB = F.interpolate(gt_RGB, size=pred_rgI.shape[2:], mode='area')
        # mask = F.interpolate(mask, size=pred_rgI.shape[2:], mode='area')
        mask = (mask > 0.99).to(torch.float32)
        if color_rep == "I":
            assert False
            # compared_gt_R_linear = gt_R.mean(dim=1, keepdim=True)
            # compared_gt_R_log = torch.log(compared_gt_R_linear.clamp(min=1e-6))
        elif color_rep == "rgI":
            I_linear = gt_RGB.mean(dim=1, keepdim=True)
            # I_log = torch.log(I_linear.clamp(min=1e-6))
            if pred_rgI.size(1) == 1:
                gt_rgI = I_linear
            else:
                rg = image_util.rgb_to_chromaticity(gt_RGB)[:, :2, :, :]
                gt_rgI = torch.cat((rg, I_linear), dim=1)
            # compared_gt_R_log = torch.cat((rg, I_log), dim=1)
        else:
            assert False, f"Not supports color_rep: {color_rep}"
        return pred_rgI, gt_rgI, gt_RGB, mask

    def forward(self, pred, data):
        assert pred["color_rep"] == "rgI"

        # GPU
        device = pred[f"pred_{self.key}"].device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Ground truth
        gt_RGB = data[f'gt_{self.key}'].to(torch.float32).requires_grad_(False)  # RGB space
        mask = (torch.min(data['mask'], dim=1, keepdim=True)[0] >= 0.999).to(torch.float32).requires_grad_(False)
        if self.key in ["S"]:
            _gt_I = gt_RGB.mean(dim=1, keepdim=True)
            mask *= (_gt_I > 1e-4) * (_gt_I < 2.0)  # ignore extreme GT shading

        # Compared pair
        color_rep = pred["color_rep"]
        pred_rgI, gt_rgI, gt_RGB, mask = self.generate_compared_pair(color_rep, pred, gt_RGB, mask)
        if pred_rgI.size(1) == 1:
            pred_rg = gt_rg = None
            pred_I, gt_I = pred_rgI, gt_rgI
        elif pred_rgI.size(1) == 3:
            pred_rg, pred_I = pred_rgI.split([2, 1], dim=1)
            gt_rg, gt_I = gt_rgI.split([2, 1], dim=1)
        else:
            assert False

        # Loss functions
        mask_rg, mask_I, mask_rgI = mask.repeat(1, 2, 1, 1), mask.repeat(1, 1, 1, 1), \
                                    mask.repeat(1, pred_rgI.size(1), 1, 1)
        if pred_rgI.size(1) == 1:
            if self.w_value > 1e-5:
                value_loss = self.value_criterion.w * (
                        self.value_criterion.I(pred_I, gt_I, mask_I) * self.value_criterion.w_I
                )
                total_loss += value_loss
            if self.w_grad > 1e-5:
                grad_loss = self.grad_criterion.w * (
                        self.grad_criterion.I(pred_I, gt_I, mask_I) * self.grad_criterion.w_I
                )
                total_loss += grad_loss
            if self.w_dssim > 1e-5:
                dssim_loss = self.w_dssim * (
                    compute_DSSIM(pred_I, gt_I, mask_I, "train", scale_invariant=False) * self.w_i
                )
                total_loss += dssim_loss
        elif pred_rgI.size(1) == 3:
            if self.w_value > 1e-5:
                value_loss = self.value_criterion.w * (
                        self.value_criterion.rg(pred_rg, gt_rg, mask_rg) * self.value_criterion.w_rg +
                        self.value_criterion.I(pred_I, gt_I, mask_I) * self.value_criterion.w_I
                )
                total_loss += value_loss
            if self.w_grad > 1e-5:
                grad_loss = self.grad_criterion.w * (
                        self.grad_criterion.rg(pred_rg, gt_rg, mask_rg) * self.grad_criterion.w_rg +
                        self.grad_criterion.I(pred_I, gt_I, mask_I) * self.grad_criterion.w_I
                )
                total_loss += grad_loss
            if self.w_dssim > 1e-5:
                dssim_loss = self.w_dssim * (
                    compute_DSSIM(pred_rg, gt_rg, mask_rg, "train", scale_invariant=False) * self.w_c +
                    compute_DSSIM(pred_I, gt_I, mask_I, "train", scale_invariant=False) * self.w_i
                )
                total_loss += dssim_loss
        else:
            assert False
        # total_loss += value_loss + grad_loss

        return total_loss


class DenseIntrinsicCriterion_RGB(nn.Module):
    Criteria = namedtuple("Criteria", ["loss", "w"])
    key = None  # should be "R" or "S"
    w_value = 1.0
    w_grad = 40.0
    w_dssim = 0.0

    def __init__(self, key, suppress_i,
                 w_value=1.0, w_grad=40.0, w_dssim=0.0):
        '''
            :param suppress_i: suppression threshold for each channel
        '''
        super(DenseIntrinsicCriterion_RGB, self).__init__()
        self.key = key
        self.w_value, self.w_grad, self.w_dssim = w_value, w_grad, w_dssim
        # p = 1.0 if self.scale_invrc_mode == "direct_intrinsics" else 0
        p = 0
        self.value_criterion = self.Criteria(
            PixelWiseLoss(p),
            self.w_value
        )
        self.grad_criterion = self.Criteria(
            MultiScaleGradientLoss(order=1, stride=2, step=4, mode="L1", eq_threshold=suppress_i),
            self.w_grad
        )
        print(f"Build DenseIntrinsicCriterion_RGB:\n"
              f"\tsuppression=({suppress_i}), \n"
              f"\tweights: ({self.w_value}, {self.w_grad}, {self.w_dssim})")

    def generate_compared_pair(self, color_rep, pred, gt_RGB, mask):
        assert color_rep == "RGB", f"color_rep should be RGB, but got {color_rep}"
        # prediction
        pred_RGB = pred[f"direct_linear_out_{self.key}"]
        # ground truth
        assert gt_RGB.shape[2:] == mask.shape[2:] == pred_RGB.shape[2:]
        mask = (mask > 0.99).to(torch.float32)
        mask = mask.min(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        mask = mask.repeat(1, pred_RGB.size(1), 1, 1)
        if pred_RGB.size(1) == 1:
            assert False, "not implemented"
            gt_RGB = gt_RGB.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        return pred_RGB, gt_RGB, mask

    def forward(self, pred, data):
        assert pred["color_rep"] == "RGB", f"color_rep should be RGB, but got {pred['color_rep']}"

        # GPU
        device = pred[f"pred_{self.key}"].device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Ground truth
        gt_RGB = data[f'gt_{self.key}'].to(torch.float32).requires_grad_(False)  # RGB space
        mask = (torch.min(data['mask'], dim=1, keepdim=True)[0] >= 0.999).to(torch.float32).requires_grad_(False)
        if self.key in ["S"]:
            _gt_I = gt_RGB.mean(dim=1, keepdim=True)
            mask *= (_gt_I > 1e-4) * (_gt_I < 2.0)  # ignore extreme GT shading

        # Compared pair
        color_rep = pred["color_rep"]
        pred_RGB, gt_RGB, mask = self.generate_compared_pair(color_rep, pred, gt_RGB, mask)
        assert pred_RGB.shape[1] in [1, 3], f"pred_RGB should have 1 or 3 channels, but got {pred_RGB.shape[1]}"

        # Loss functions
        if self.w_value > 1e-5:
            value_loss = self.value_criterion.w * (
                    self.value_criterion.loss(pred_RGB, gt_RGB, mask)
            )
            total_loss += value_loss
        if self.w_grad > 1e-5:
            grad_loss = self.grad_criterion.w * (
                    self.grad_criterion.loss(pred_RGB, gt_RGB, mask)
            )
            total_loss += grad_loss
        if self.w_dssim > 1e-5:
            dssim_loss = self.w_dssim * (
                    compute_DSSIM(pred_RGB, gt_RGB, mask, "train", scale_invariant=False)
            )
            total_loss += dssim_loss
        return total_loss


class IIWCriterion(nn.Module):
    w: float
    w_ineq: float
    margin_eq: float
    margin_ineq: float

    def __init__(self, w, w_ineq, margin_eq, margin_ineq):
        super(IIWCriterion, self).__init__()
        self.w = w
        self.w_ineq = w_ineq
        self.margin_eq = margin_eq
        self.margin_ineq = margin_ineq
        print(f"Build IIWCriterion with w={self.w}, w_ineq={self.w_ineq}, "
              f"margin_eq = {self.margin_eq}, margin_ineq= {self.margin_ineq}")

    def BatchRankingLoss(self, prediction_R, judgements_eq, judgements_ineq, random_flip):
        device = prediction_R.device
        eq_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        ineq_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        eq_weight = torch.tensor(0.0, dtype=torch.float32, device=device)
        ineq_weight = torch.tensor(0.0, dtype=torch.float32, device=device)

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)
        num_channel = prediction_R.size(0)

        # evaluate equality annotations densely
        if judgements_eq.size(1) > 2:
            R_vec = prediction_R.view(num_channel, -1)

            y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            y_2 = torch.floor(judgements_eq[:,2] * rows).long()
            if random_flip:
                x_1 = cols - 1 - torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_eq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = torch.floor(judgements_eq[:,3] * cols).long()

            point_1_idx_linaer = y_1 * cols + x_1
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec, 1, point_1_idx_linaer)
            points_2_vec = torch.index_select(R_vec, 1, point_2_idx_linear)

            weight = judgements_eq[:, 4]

            # compute loss
            # eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.pow(points_1_vec - points_2_vec,2),0) ))
            # num_valid_eq += judgements_eq.size(0)
            eq_loss = F.relu((points_1_vec - points_2_vec).abs().mean(dim=0) - self.margin_eq, True) * weight
            eq_loss = eq_loss.sum()/weight.sum().clamp(min=1e-6)
            # eq_weight = weight.sum()

        # compute inequality annotations
        if judgements_ineq.size(1) > 2:
            R_intensity = torch.mean(prediction_R, 0)
            R_vec_mean = R_intensity.view(1, -1)

            y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            y_2 = torch.floor(judgements_ineq[:,2] * rows).long()

            if random_flip:
                x_1 = cols - 1 - torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_ineq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            point_1_idx_linaer = y_1 * cols + x_1
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec_mean, 1, point_1_idx_linaer).squeeze(0)
            points_2_vec = torch.index_select(R_vec_mean, 1, point_2_idx_linear).squeeze(0)
            weight = judgements_ineq[:, 4]

            # point 2 should be always darker than (<) point 1
            # compute loss
            # ineq_loss = torch.sum(torch.mul(weight, torch.pow( relu_layer(points_2_vec - points_1_vec + tau),2)  ) )
            # num_included = torch.sum(torch.ge(points_2_vec - points_1_vec, -tau).float())
            # num_valid_ineq += num_included
            ineq_loss = F.relu(points_2_vec - points_1_vec + self.margin_ineq, True) * weight
            ineq_loss = ineq_loss.sum()/weight.sum().clamp(min=1e-6)
            # ineq_weight = weight.sum()

        # avoid divide by zero
        # return eq_loss/max(num_valid_eq, 1e-6) + ineq_loss/max(num_valid_ineq, 1e-6)
        # avg_loss = (eq_loss + ineq_loss)/(eq_weight + ineq_weight).clamp(min=1e-6)
        avg_loss = (eq_loss + self.w_ineq * ineq_loss) / (1 + self.w_ineq)
        return avg_loss

    def forward(self, pred, data, **params):
        color_rep = pred["color_rep"]
        assert color_rep in ["rgI", "RGB"]
        # pred_rgI = pred["direct_linear_out_R"]

        # GPU
        device = pred["pred_R"].device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        targets = data["targets"]

        num_imgs = pred["pred_R"].size(0)
        for i in range(0, num_imgs):
            # o_h, o_w = targets["original_h"][i], targets["original_w"][i]
            judgements_eq = targets["eq_mat"][i].to(device).requires_grad_(False)
            judgements_ineq = targets["ineq_mat"][i].to(device).requires_grad_(False)
            random_flip = targets["random_flip"][i]
            if color_rep == "rgI":
                intensity = pred["direct_linear_out_R"][i, 2:3, :, :]
            elif color_rep == "RGB":
                intensity = pred["direct_linear_out_R"][i, :, :, :].mean(dim=0, keepdim=True)
            else:
                raise NotImplementedError("Unknown color representation: {}".format(color_rep))
            total_loss += self.BatchRankingLoss(intensity, judgements_eq, judgements_ineq, random_flip)
        total_loss = total_loss/num_imgs
        return total_loss * self.w
