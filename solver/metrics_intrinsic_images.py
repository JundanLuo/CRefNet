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


from collections import namedtuple

# from solver.ssim import ssim_loss
from kornia.losses import ssim_loss
from utils.average_meter import AverageMeter


def scale_matching(pred, target, mask):
    assert pred.ndim == target.ndim == 4, "Only supports tensor: [B, C, H, W]"
    assert pred.shape == target.shape == mask.shape, \
        f"Different size: pred {pred.shape}, target {target.shape}, mask {mask.shape}"
    pred = pred.mul(mask)
    target = target.mul(mask)
    alpha = (target * pred).sum(dim=[1, 2, 3]) / (pred ** 2).sum(dim=[1, 2, 3]).clamp(min=1e-6)
    alpha = alpha.mul((pred ** 2).sum(dim=[1, 2, 3]) > 1e-5).reshape([pred.size(0), 1, 1, 1])
    return alpha


def scale_invariant_MSE(pred, target, mask):
    assert pred.shape == target.shape == mask.shape, \
        f"Different size: pred {pred.shape}, target {target.shape}, mask {mask.shape}"
    assert pred.ndim == 4, "Only supports tensor: [B, C, H, W]"
    alpha = scale_matching(pred, target, mask)
    pred = pred.mul(mask).mul(alpha)
    target = target.mul(mask)
    loss = ((pred - target) ** 2).sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3]).clamp(min=1e-6)
    loss = loss.mean()
    return loss


def scale_invariant_LMSE(pred, target, mask):
    assert pred.ndim == target.ndim == 4, "Only supports tensor: [B, C, H, W]"
    H, W = pred.shape[2:]
    window_size = int(max(pred.shape) * 0.1 + 0.5)
    window_shift = int(window_size / 2.0 + 0.5)
    loss = 0.0
    cnt = 0
    for i in range(0, H - window_shift + 1, window_shift):
        for j in range(0, W - window_shift + 1, window_shift):
            cnt += 1
            ii = min(H, i + window_size)
            jj = min(W, j + window_size)
            target_curr = target[:, :, i:ii, j:jj]
            pred_curr = pred[:, :, i:ii, j:jj]
            mask_curr = mask[:, :, i:ii, j:jj]
            loss += scale_invariant_MSE(pred_curr, target_curr, mask_curr)
    loss /= cnt
    return loss


def compute_DSSIM(pred, target, mask, mode, scale_invariant=True):
    pred = pred.mul(mask)
    target = target.mul(mask)
    if scale_invariant:
        alpha = scale_matching(pred, target, mask)
    else:
        alpha = 1.0
    loss = ssim_loss(pred.mul(alpha), target,
                     window_size=11, max_val=1.0,
                     reduction="mean",
                     padding="valid")

    if mode == "test":
        assert pred.size(0) == target.size(0) == mask.size(0) == 1, "Batch_size must be 1"
        for i in range(2, 11):
            for j in [-1, 1]:
                step = j * 2 ** (-i)
                while 0 < (alpha + step).item() < 5:
                    new_alpha = alpha + step
                    new_loss = ssim_loss(pred.mul(new_alpha), target,
                                         window_size=11, max_val=1.0,
                                         reduction="mean",
                                         padding="valid")
                    if new_loss < loss:
                        alpha = new_alpha
                        loss = new_loss
                    else:
                        break
    return loss


class SI_IntrinsicImageMetricsMeter(object):

    supported_DSSIM_mode = ["test", "val"]
    Result = namedtuple("Result", ["si_MSE", "si_LMSE", "DSSIM"])

    def __init__(self, name, compute_siMSE=False, compute_siLMSE=False, compute_DSSIM=False, mode_DSSIM="test"):
        self.name = name
        self.compute_siMSE = compute_siMSE
        self.compute_siLMSE = compute_siLMSE
        self.compute_DSSIM = compute_DSSIM
        if mode_DSSIM not in self.supported_DSSIM_mode:
            raise Exception(f"Not supports mode: {mode_DSSIM}")
        self.mode_DSSIM = mode_DSSIM
        self.reset()

    def reset(self):
        self.si_MSE_meter = AverageMeter(name="si_MSE")
        self.si_LMSE_meter = AverageMeter(name="si_LMSE")
        self.DSSIM_meter = AverageMeter(name="DSSIM")

    def update(self, pred, target, mask):
        assert pred.ndim == target.ndim == 4, "Only supports tensor: [B, C, H, W]"
        batch_size = pred.size(0)
        if self.compute_siMSE:
            self.si_MSE_meter.update(scale_invariant_MSE(pred, target, mask), n=batch_size)
        if self.compute_siLMSE:
            self.si_LMSE_meter.update(scale_invariant_LMSE(pred, target, mask), n=batch_size)
        if self.compute_DSSIM:
            if self.mode_DSSIM == "test":
                for i in range(batch_size):
                    self.DSSIM_meter.update(compute_DSSIM(pred[i:i+1], target[i:i+1], mask[i:i+1], self.mode_DSSIM),
                                            n=1)
            elif self.mode_DSSIM == "val":
                self.DSSIM_meter.update(compute_DSSIM(pred, target, mask, self.mode_DSSIM),
                                        n=batch_size)
            else:
                raise Exception(f"Not supports mode: {self.mode_DSSIM}")

    def get_results(self) -> Result:
        return self.Result(si_MSE=self.si_MSE_meter.avg, si_LMSE=self.si_LMSE_meter.avg, DSSIM=self.DSSIM_meter.avg)
