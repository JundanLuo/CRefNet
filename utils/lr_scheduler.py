import torch
from yacs.config import CfgNode as CN
from timm.scheduler.scheduler import Scheduler


def build_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, cfg: CN):
    if scheduler_type == 'SGDR':
        assert False, "Not implemented"
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
        #                                                                       cfg.TRAIN.T_0, cfg.TRAIN.T_mult,
        #                                                                       cfg.TRAIN.min_lr, -1, False)
        # self.logtool.log_text(f'Scheduler',
        #                       f'CosineAnnealingWarmRestarts: '
        #                       f'T_0:{cfg.TRAIN.T_0:.6f}, T_mult:{cfg.TRAIN.T_mult:.6f}, '
        #                       f'min_lr:{cfg.TRAIN.min_lr:.6f}',
        #                       global_step=0)
    elif scheduler_type == "StepLR":
        assert False, "Not implemented"
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=cfg.TRAIN.T_0,
        #                                                  gamma=cfg.TRAIN.gamma)
        # self.logtool.log_text(f'Scheduler',
        #                       f'\tStepLR: '
        #                       f'step_size:{cfg.TRAIN.T_0:.2f}, gamma: {cfg.TRAIN.gamma:.3f}',
        #                       global_step=0)
    elif scheduler_type == "StepLRWarmUp":
        from timm.scheduler.step_lr import StepLRScheduler
        decay_iters = cfg.TRAIN.LR_SCHEDULER.decay_epoch * cfg.TRAIN.epoch_iters
        warmup_iters = cfg.TRAIN.LR_SCHEDULER.warmup_epoch * cfg.TRAIN.epoch_iters
        scheduler = StepLRScheduler(optimizer,
                                    decay_t=decay_iters, decay_rate=cfg.TRAIN.LR_SCHEDULER.decay_rate,
                                    warmup_t=warmup_iters, warmup_lr_init=cfg.TRAIN.LR_SCHEDULER.warmup_init_lr,
                                    t_in_epochs=False)
    elif scheduler_type == "Linear":
        num_iters = cfg.TRAIN.LR_SCHEDULER.decay_epoch * cfg.TRAIN.epoch_iters
        warmup_iters = cfg.TRAIN.LR_SCHEDULER.warmup_epoch * cfg.TRAIN.epoch_iters
        scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_iters,
            lr_min_rate=cfg.TRAIN.LR_SCHEDULER.decay_rate,
            warmup_lr_init=cfg.TRAIN.LR_SCHEDULER.warmup_init_lr,
            warmup_t=warmup_iters,
            t_in_epochs=False, )
    elif scheduler_type in [None, False]:
        assert False, "Not implemented"
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=9999, gamma=0)
    else:
        raise Exception(f"Error: not support scheduler type: {cfg.TRAIN.scheduler}")
    return scheduler


# Modified from:
# https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/lr_scheduler.py#L66
class LinearLRScheduler(Scheduler):
    # --------------------------------------------------------
    # Swin Transformer
    # Copyright (c) 2021 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ze Liu
    # --------------------------------------------------------

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [max(v - ((v - v * self.lr_min_rate) * (t / total_t)), v * self.lr_min_rate) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None