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


from math import pi, cos

import torch
from yacs.config import CfgNode as CN

from modeling.basic_net import BasicNet
from utils.lr_scheduler import build_scheduler
import constants as C


class TrainingConfigManager(object):
    SUPPORT_TRAINING = ["RE", "with_RR", "SE", "with_SR"]

    class Option(object):
        train_re = None
        train_with_rr = None
        train_se = None
        train_with_sr = None
        recstr_cosine_cycle = None  # weight decay cycle for training the reconstruction branch
        scheduler_name = None
        # scheduler_delay = None

    class Weights(object):
        re = 1.0
        rr = 0.0
        se = 1.0
        sr = 0.0

    op = None
    w = Weights()
    optimizer = None
    scheduler = None
    criterion = None
    iterator_train = None
    rm_cgi_epoch = -1

    def __init__(self, cfg: CN, criterion, data_loaders):
        # training strategy
        assert cfg.TRAIN.STRATEGY.type in self.SUPPORT_TRAINING
        self.op = self.Option()
        self.op.train_re = cfg.TRAIN.STRATEGY.type in ["RE", "with_RR"]  # train RE branch
        self.op.train_with_rr = cfg.TRAIN.STRATEGY.type in ["with_RR"]  # train RR branch
        self.op.train_se = (cfg.TRAIN.STRATEGY.type in ["SE", "with_SR"])  # train SE branch
        self.op.train_with_sr = (cfg.TRAIN.STRATEGY.type in ["with_SR"])  # train SR branch
        self.op.recstr_cosine_cycle = cfg.TRAIN.STRATEGY.cosine_cycle

        # criterion
        self.criterion = criterion

        # data loaders
        self.iterator_train = iter(data_loaders)
        if cfg.TRAIN.dataset == C.MPI_Sintel:
            self.rm_cgi_epoch = cfg.TRAIN.STRATEGY.rm_cgi_epoch
        else:
            self.rm_cgi_epoch = -1

    def config_optimizer_and_scheduler(self, device, model: BasicNet, cfg: CN):
        # move criterion on device
        _criterion = self.criterion
        self.criterion = {}
        for k, v in _criterion.items():
            self.criterion[k] = {}
            for kc, c in v.items():
                self.criterion[k][kc] = c.to(device)

        # optimizer and scheduler
        optim_params = model.configure_params_for_optimizer(cfg)
        self.optimizer = None
        if cfg.TRAIN.optim == "Adam":
            self.optimizer = torch.optim.Adam(optim_params, lr=cfg.TRAIN.lr, betas=(0.9, 0.999),
                                              weight_decay=cfg.TRAIN.weight_decay)
        elif cfg.TRAIN.optim == "AdamW":
            self.optimizer = torch.optim.AdamW(optim_params, lr=cfg.TRAIN.lr, betas=(0.9, 0.999),
                                               weight_decay=cfg.TRAIN.weight_decay)
        else:
            raise Exception("Doesn't support the optimizer type: %s" % cfg.TRAIN.optim)
        print(f"Use {cfg.TRAIN.optim} optimizer.")

        self.op.scheduler_name = cfg.TRAIN.LR_SCHEDULER.name
        self.scheduler = build_scheduler(self.op.scheduler_name, self.optimizer, cfg)
        self.optimizer.zero_grad()  # this zero gradient update is needed to avoid a warning message

    def update_bf_epoch(self, epoch):
        # Data loaders
        if epoch == self.rm_cgi_epoch:
            self.iterator_train.remove(C.CGIntrinsics)

        # branch weights
        if self.op.train_with_rr:
            self.w.rr = self.cosine_annealing_schedule(epoch, self.op.recstr_cosine_cycle, 10, 0.1)
        if self.op.train_with_sr:
            self.w.sr = self.cosine_annealing_schedule(epoch, self.op.recstr_cosine_cycle, 1.0, 0.1)

    def update_af_epoch(self, epoch):
        # update learning rate
        if self.op.scheduler_name == "StepLR":
            self.scheduler.step()
        # if cfg.TRAIN.scheduler == 'plateau':
        #     scheduler.step(metrics=rs.rel_eval)
        # else:
        #     raise Exception(f"Error: not support scheduler type: {cfg.TRAIN.scheduler}")

    def update_bf_iteration(self, epoch, curr_iter, epoch_iters):
        # update learning rate
        if self.op.scheduler_name in ["StepLRWarmUp", "Linear"]:
            self.scheduler.step_update(epoch * epoch_iters + curr_iter)
        # if self.op.scheduler_type == 'SGDR' and epoch >= self.op.scheduler_delay:
        #     self.scheduler.step(epoch - self.op.scheduler_delay + i / epoch_iters)

    def current_lr(self):
        # return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def cosine_annealing_schedule(epoch, epochs_per_cycle, lrate_max, lrate_min):
        if epoch < epochs_per_cycle:
            cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
            out = max(lrate_max/2 * (cos(cos_inner) + 1), lrate_min)
        else:
            out = lrate_min
        return out