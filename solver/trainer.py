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


import os
import time

import torch
from yacs.config import CfgNode as CN

import evaluate
from utils import model_loader, image_util
from utils.average_meter import AverageMeter
from solver import metrics_intrinsic_images
import constants as C
from solver.train_manager import TrainingConfigManager


class ValResult(object):
    rel_eval: float
    WHDR_rgb: float
    WHDR_srgb: float
    WHDR_eq_rgb: float
    WHDR_ineq_rgb: float
    WHDR_eq_srgb: float
    WHDR_ineq_srgb: float
    AP: float
    DSSIM: float
    si_MSE: float
    si_LMSE: float
    NCC_same_scene: float
    epoch: int

    def __init__(self, rel_eval=0.0,
                 WHDR_rgb=1.0, WHDR_srgb=1.0,
                 WHDR_eq_rgb=1.0, WHDR_ineq_rgb=1.0, WHDR_eq_srgb=1.0, WHDR_ineq_srgb=1.0,
                 AP=0.0,
                 DSSIM=1.0, si_MSE=999999, si_LMSE=999999,
                 NCC_same_scene=0.0,
                 epoch=-1):
        self.rel_eval = rel_eval
        self.WHDR_rgb = WHDR_rgb
        self.WHDR_srgb = WHDR_srgb
        self.WHDR_eq_rgb, self.WHDR_ineq_rgb = WHDR_eq_rgb, WHDR_ineq_rgb
        self.WHDR_eq_srgb, self.WHDR_ineq_srgb = WHDR_eq_srgb, WHDR_ineq_srgb
        self.AP = AP
        self.DSSIM = DSSIM
        self.si_MSE = si_MSE
        self.si_LMSE = si_LMSE
        self.NCC_same_scene = NCC_same_scene
        self.epoch = epoch

    def __str__(self):
        return f"rel_eval {self.rel_eval: .3f}, " \
            f"WHDR(rgb) {self.WHDR_rgb:.4f}, WHDR(srgb) {self.WHDR_srgb:.4f}, " \
            f"WHDR_eq(rgb) {self.WHDR_eq_rgb: .4f}, WHDR_ineq(rgb) {self.WHDR_ineq_rgb: .4f}, " \
            f"WHDR_eq(srgb) {self.WHDR_eq_srgb: .4f}, WHDR_ineq(srgb) {self.WHDR_ineq_srgb: .4f}, " \
            f"AP {self.AP: .4f}, " \
            f"si_MSE {self.si_MSE:.4f}, si_LMSE {self.si_LMSE:.4f}, DSSIM {self.DSSIM:.4f}, " \
            f"NCC(same scene){self.NCC_same_scene:.4f}"


class LoggingTool(object):
    # logger
    log = None
    # tensorboard writers
    writer_text = None
    writer_img = None
    # plots and images
    plots = {}
    used_plot_titles = set()

    class PlotInfo(object):
        def __init__(self, title, step):
            self.title = title
            self.step = step

    def __init__(self, log=None, writer_text=None, writer_img=None):
        self.log = log
        self.writer_text = writer_text
        self.writer_img = writer_img

    def log_text(self, title, text, writer_logging=True, **params):
        self.log.info(f'{title}:\n'
                      f'{text}')
        if writer_logging:
            self.writer_text.add_text(f"{title}",
                                      text,
                                      global_step=params["global_step"])

    def init_plot(self, key, plot_title):
        # if key in self.plots.keys():
        #     assert False, f"Already exists figure {key}: {plot_title}"
        if plot_title in self.used_plot_titles:
            assert False, f"Already inits plot_title: {plot_title}"
        self.plots[key] = self.PlotInfo(plot_title, 0)
        self.used_plot_titles.add(plot_title)

    def plot_scalar_value(self, key, v):
        pi = self.plots[key]
        self.writer_text.add_scalar(pi.title, v,
                                    global_step=pi.step)
        self.plots[key].step += 1

    def plot_image(self, key, v):
        pi = self.plots[key]
        self.writer_img.add_image(pi.title, v, global_step=pi.step, dataformats="CHW")
        self.plots[key].step += 1


class Trainer(object):
    logtool = None
    device = None
    model = None
    train_manager = None
    tg_WHDR = 0.16
    tg_DSSIM = 0.1
    tg_NCC_same_scene = 1.0
    tg_AP = 98.0

    def __init__(self):
        pass

    def configure_device_and_model(self, cfg: CN, model):
        self.model = model

        # device
        if cfg.MODEL.use_gpu:
            if not torch.cuda.is_available():
                raise Exception("Cuda is not available!")
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # model
        if cfg.MODEL.checkpoint is not None:
            model_loader.load_model(self.model, cfg.MODEL.checkpoint, strict=True)
        # Data parallel
        if torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUs.")
            from modeling.basic_net import DataParallelWrapper
            self.model = DataParallelWrapper(self.model)
        self.model = self.model.to(self.device)

    def train(self, cfg: CN, model, train_manager: TrainingConfigManager, _writers, _log):
        # Logging tool
        self.logtool = LoggingTool(_log, _writers["text"], _writers["image"])
        self.logtool.log_text("Configuration", str(cfg), global_step=0)

        # Configure: device, model, optimizer, scheduler
        self.train_manager = train_manager
        self.configure_device_and_model(cfg, model)
        self.train_manager.config_optimizer_and_scheduler(self.device, self.model, cfg)

        # Initialize
        best_val_result = ValResult(rel_eval=0)
        self.logtool.init_plot("lr", "Train/learning_rate(epoch)")
        self.logtool.init_plot("lr_iter", "Train/learning_rate(iter)")
        self.logtool.init_plot("loss", "Train/loss")
        self.logtool.init_plot("loss_re", "Train/loss_re")
        self.logtool.init_plot("loss_rr", "Train/loss_rr")
        self.logtool.init_plot("loss_se", "Train/loss_se")
        self.logtool.init_plot("loss_sr", "Train/loss_sr")
        self.logtool.init_plot("wre", "Train/weight_re")
        self.logtool.init_plot("wrr", "Train/weight_rr")
        self.logtool.init_plot("wse", "Train/weight_se")
        self.logtool.init_plot("wsr", "Train/weight_sr")
        self.logtool.log_text("Training epochs",
                              f"\tNumber of training epochs: {cfg.TRAIN.num_epoch}, "
                              f"\tIterations for each epoch: {cfg.TRAIN.epoch_iters}",
                              global_step=0)
        assert isinstance(cfg.TRAIN.epoch_iters, int) and cfg.TRAIN.epoch_iters > 0, \
            f" Erroneous iterations for each epoch: {cfg.TRAIN.epoch_iters}, " \
            f"'TRAIN.epoch_iters' must be a positive integer."


        # Train
        for epoch in range(0, cfg.TRAIN.num_epoch):
            # update training settings before each epoch
            self.train_manager.update_bf_epoch(epoch)

            # unfreeze model
            self.model.unfreeze()

            # log learning rate and weights
            self.logtool.plot_scalar_value("lr", self.train_manager.current_lr())
            self.logtool.log_text("Epoch",
                                  f"\tEpoch {epoch}"
                                  f"\n\ttraining datasets: {self.train_manager.iterator_train.names}"
                                  f"\n\tweights for re: {self.train_manager.w.re}, rr: {self.train_manager.w.rr}, "
                                  f"se: {self.train_manager.w.se}, sr: {self.train_manager.w.sr}",
                                  global_step=epoch)
            self.logtool.plot_scalar_value("wre", self.train_manager.w.re)
            self.logtool.plot_scalar_value("wrr", self.train_manager.w.rr)
            self.logtool.plot_scalar_value("wse", self.train_manager.w.se)
            self.logtool.plot_scalar_value("wsr", self.train_manager.w.sr)

            # Train for one epoch
            end = time.time()
            loss_meter = AverageMeter("total_loss")
            loss_re_meter = AverageMeter("loss_re")
            loss_rr_meter = AverageMeter("loss_rr")
            loss_se_meter = AverageMeter("loss_se")
            loss_sr_meter = AverageMeter("loss_sr")
            for i in range(0, cfg.TRAIN.epoch_iters):
                # Update train settings before each iteration
                self.train_manager.update_bf_iteration(epoch, i, cfg.TRAIN.epoch_iters)
                # Train for one iteration
                data_train = next(self.train_manager.iterator_train)  # Samples the batch
                train_out = self.training_step(data_train)
                del data_train
                # Update loss meters
                loss_meter.update(train_out["total_loss"], train_out["batch_size"])
                loss_re_meter.update(train_out["loss_re"], train_out["batch_size"])
                loss_rr_meter.update(train_out["loss_rr"], train_out["batch_size"])
                loss_se_meter.update(train_out["loss_se"], train_out["batch_size"])
                loss_sr_meter.update(train_out["loss_sr"], train_out["batch_size"])
                # Log and visualization
                if (i + 1) % cfg.TRAIN.disp_iters == 0:
                    # log learning rate
                    if cfg.TRAIN.LR_SCHEDULER.name in ['StepLRWarmUp', 'Linear']:
                        self.logtool.plot_scalar_value("lr_iter", self.train_manager.current_lr())
                        print(f"Current lr: {self.train_manager.current_lr()}")
                    # log training loss
                    avg_time = (time.time() - end) / cfg.TRAIN.disp_iters
                    self.logtool.log_text("Training loss",
                                          f"\tepoch {epoch}, iteration {i}/{cfg.TRAIN.epoch_iters-1}, "
                                          f"loss {loss_meter.avg:.3f}, "
                                          f"time: {avg_time:.3f}",
                                          writer_logging=False)
                    self.logtool.plot_scalar_value("loss", loss_meter.avg)
                    self.logtool.plot_scalar_value("loss_re", loss_re_meter.avg)
                    self.logtool.plot_scalar_value("loss_rr", loss_rr_meter.avg)
                    self.logtool.plot_scalar_value("loss_se", loss_se_meter.avg)
                    self.logtool.plot_scalar_value("loss_sr", loss_sr_meter.avg)
                    loss_meter.reset()
                    loss_re_meter.reset()
                    loss_rr_meter.reset()
                    loss_se_meter.reset()
                    loss_sr_meter.reset()

                    if cfg.TRAIN.visualize and (i + 1) % (cfg.TRAIN.disp_iters * 3) == 0:
                        self.logtool.init_plot("img",
                                               f"Train/{train_out['dataset']}/train_{epoch:03d}_{i:03d}")
                        img_grids = self.visualize_data_pred(train_out["data_pred_dicts"], train_out["batch_size"])
                        for (d_idx, vis_grid) in img_grids:
                            if cfg.MODE == "release":
                                self.logtool.plot_image("img", vis_grid)
                            else:
                                image_util.save_srgb_image(vis_grid, os.path.join(cfg.DIR, "train", "samples"),
                                                           f"train_{epoch:03d}_{i:03d}_{d_idx:03d}.png")
                    end = time.time()

            # Validate after one epoch
            if cfg.MODE == "debug":
                rs = ValResult(rel_eval=0.0, epoch=epoch)
            else:
                if cfg.TRAIN.dataset in [C.CGIntrinsics, C.IIW]:
                    if self.train_manager.op.train_re:
                        rs = self.validate(epoch, cfg, val_ordinal=C.IIW, main_metrics="ordinal")
                    elif self.train_manager.op.train_se:
                        rs = self.validate(epoch, cfg, val_s_smooth=C.SAW, main_metrics="s_smooth")
                    else:
                        assert False
                elif cfg.TRAIN.dataset in [C.MPI_Sintel, C.MIT_Intrinsic]:
                    rs = self.validate(epoch, cfg, val_dense=cfg.TRAIN.dataset, main_metrics="dense")
                else:
                    assert f"Not supports training dataset {cfg.TRAIN.dataset}!"
            if rs.rel_eval > best_val_result.rel_eval:
                best_val_result = rs
            self.logtool.log_text(f'Validation',
                                  f"Epoch {rs.epoch}:\n"
                                  f'\t{rs}\n'
                                  f'best epoch: {best_val_result.epoch}\n'
                                  f'\t{best_val_result}\n',
                                  global_step=epoch)

            # Save model
            self.save_model(os.path.join(cfg.DIR, "train/"), epoch, cfg.TRAIN.save_per_epoch,
                            rs, best_val_result, cfg)

            # Update train settings after each epoch
            self.train_manager.update_af_epoch(epoch)

    def validate(self, epoch, cfg, val_ordinal=None, val_s_smooth=None, val_dense=None, val_time_lapse=None,
                 main_metrics="ordinal"):
        assert main_metrics in ["ordinal", "dense", "time_lapse", "s_smooth"]

        # freeze model
        self.model.freeze()

        writer = self.logtool.writer_text
        writer_img = self.logtool.writer_img

        # validate
        rs = ValResult(epoch=epoch)
        rel_eval_ordinal = rel_eval_s_smooth = rel_eval_dense = rel_eval_time_lapse = None
        if val_ordinal is not None:
            assert val_ordinal in [C.IIW]
            visualize_dir = os.path.join(cfg.DIR, "train/validate/iiw") if cfg.VAL.visualize else None
            result_rgb, result_srgb = evaluate.validate_iiw(self.model, self.device,
                                                            cfg, "val",
                                                            False,
                                                            writer=writer_img if cfg.VAL.visualize else None,  #and main_dataset == "IIW" else None,
                                                            visualize_dir=visualize_dir,
                                                            label=f'val_iiw_{epoch:03d}')
            rs.WHDR_rgb, rs.WHDR_eq_rgb, rs.WHDR_ineq_rgb = result_rgb.WHDR, result_rgb.WHDR_eq, result_rgb.WHDR_ineq
            rs.WHDR_srgb, rs.WHDR_eq_srgb, rs.WHDR_ineq_srgb = result_srgb.WHDR, result_srgb.WHDR_eq, result_srgb.WHDR_ineq
            writer.add_scalar("Val/WHDR(rgb)", rs.WHDR_rgb, global_step=epoch)
            writer.add_scalar("Val/WHDR_eq(rgb)", rs.WHDR_eq_rgb, global_step=epoch)
            writer.add_scalar("Val/WHDR_ineq(rgb)", rs.WHDR_ineq_rgb, global_step=epoch)
            writer.add_scalar("Val/WHDR(srgb)", rs.WHDR_srgb, global_step=epoch)
            writer.add_scalar("Val/WHDR_eq(srgb)", rs.WHDR_eq_srgb, global_step=epoch)
            writer.add_scalar("Val/WHDR_ineq(srgb)", rs.WHDR_ineq_srgb, global_step=epoch)
            rel_eval_ordinal = self.tg_WHDR / rs.WHDR_srgb
        if val_s_smooth is not None:
            assert val_s_smooth in [C.SAW]
            visualize_dir = os.path.join(cfg.DIR, "train/validate/saw") if cfg.VAL.visualize else None
            AP, _, _ = evaluate.validate_saw(self.model, self.device,
                                          cfg, "val",
                                          True,
                                          True,
                                          writer=writer_img if cfg.VAL.visualize else None,
                                          visualize_dir=visualize_dir,
                                          label=f'val_saw_{epoch:03d}')
            rs.AP = AP
            writer.add_scalar("Val/AP(c)", rs.AP, global_step=epoch)
            rel_eval_s_smooth = rs.AP / self.tg_AP
        if val_dense is not None:
            assert val_dense in [C.CGIntrinsics, C.MPI_Sintel, C.MIT_Intrinsic]
            visualize_dir = os.path.join(cfg.DIR, f"train/validate/{val_dense}") if cfg.VAL.visualize else None
            writers = {
                "text": writer,
                "image": writer_img if cfg.VAL.visualize else None,
            }
            split = {
                C.CGIntrinsics: None,
                C.MPI_Sintel: cfg.TRAIN.split,
                C.MIT_Intrinsic: None,
            }[val_dense]
            # mode = {
            #     "CGI": "val",
            #     "MPI": "test"
            # }[val_dense]
            si_MSE, si_LMSE, DSSIM = evaluate.validate_dense(val_dense, split,
                                                             self.model, self.device,
                                                             cfg, "val",
                                                             val="R" if self.train_manager.op.train_re else "S",
                                                             display_process=False,
                                                             writers=writers,
                                                             visualize_dir=visualize_dir,
                                                             epoch=epoch,
                                                             label=f'val_{val_dense}_{epoch:03d}'
                                                             )
            rs.si_MSE = si_MSE
            rs.si_LMSE = si_LMSE
            rs.DSSIM = DSSIM
            writer.add_scalar("Val/si_MSE", rs.si_MSE, global_step=epoch)
            writer.add_scalar("Val/si_LMSE", rs.si_LMSE, global_step=epoch)
            writer.add_scalar("Val/DSSIM",  rs.DSSIM,  global_step=epoch)
            rel_eval_dense = self.tg_DSSIM / rs.DSSIM

        rs.rel_eval = {
            "ordinal": rel_eval_ordinal,
            "s_smooth": rel_eval_s_smooth,
            "dense": rel_eval_dense,
            # "time_lapse": rel_eval_time_lapse
        }[main_metrics]
        writer.add_scalar("Val/relative_result", rs.rel_eval, global_step=epoch)
        return rs

    def training_step(self, _data):
        dataset = _data["dataset"][0]
        if dataset in [C.CGIntrinsics, C.Hypersim, C.IIW, C.MPI_Sintel, C.MIT_Intrinsic]:
            # assert batch_size == _data["srgb_img"].size(0)
            train_re = self.train_manager.op.train_re  # train reflectance estimation
            train_rr = self.train_manager.op.train_with_rr and dataset in \
                       [C.CGIntrinsics, C.Hypersim, C.MPI_Sintel, C.MIT_Intrinsic]  # train reflectance reconstruction
            train_se = self.train_manager.op.train_se  # train shading estimation
            train_sr = self.train_manager.op.train_with_sr and dataset in \
                       [C.CGIntrinsics, C.Hypersim, C.MPI_Sintel, C.MIT_Intrinsic]  # train shading reconstruction
            if train_rr:
                assert self.model.arch_name in ["CRefNet", "CRefNet_R", "CRefNet_Fully_ResBlocks",
                                                "CRefNet_Swin_Encoder"], \
                    f"arch {self.model.arch_name} not supports training with the RR-branch."
            data = {}
            for k, v in _data.items():
                if torch.is_tensor(v):
                    data[k] = v.to(self.device).requires_grad_(False)
                else:
                    data[k] = v

            # forward
            loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            data_pred_dicts = {}
            # reflectance estimation
            if train_re:
                data_mode_re, pred_mode_re, loss_mode_re = self.train_one_reflectance_estimation_step(dataset, data)
                loss = loss + loss_mode_re * self.train_manager.w.re
                data_pred_dicts["RE"] = (data_mode_re, pred_mode_re)

            # reflectance reconstruction
            if train_rr:
                data_mode_ = data.copy()
                for k in ["gt_R", "srgb_img", "rgb_img", "mask"]:
                    data_mode_[k] = data_mode_[k].detach().clone()
                data_mode_rr = data_mode_.copy()
                data_mode_rr["srgb_img"] = data_mode_rr["rgb_img"] = data_mode_rr["gt_R"]
                pred_mode_rr = self.model(data_mode_rr["srgb_img"], mode="RR")
                loss_mode_rr = self.train_manager.criterion["R"]["dense"](pred_mode_rr, data_mode_rr)
                loss = loss + loss_mode_rr * self.train_manager.w.rr
                data_pred_dicts["RR"] = (data_mode_rr, pred_mode_rr)

            # shading estimation
            if train_se:
                data_mode_se = data.copy()
                pred_mode_se = self.model(data_mode_se["srgb_img"], mode="SE")
                loss_mode_se = self.train_manager.criterion["S"]["dense"](pred_mode_se, data_mode_se)
                loss = loss + loss_mode_se * self.train_manager.w.se
                # if dataset == "IIW":
                #     data_mode_se["gt_R"] = data_mode_se["srgb_img"]
                data_pred_dicts["SE"] = (data_mode_se, pred_mode_se)

            # shading reconstruction
            if train_sr:
                data_mode_sr = data.copy()
                data_mode_sr["srgb_img"] = data_mode_sr["rgb_img"] = data_mode_sr["gt_S"]
                pred_mode_sr = self.model(data_mode_sr["srgb_img"],
                                          mode="SR")
                # loss sr
                loss_mode_sr = self.train_manager.criterion["S"]["dense"](pred_mode_sr, data_mode_sr)
                loss = loss + loss_mode_sr * self.train_manager.w.sr
                # if dataset == "IIW":
                #     data_mode_se["gt_R"] = data_mode_sr["gt_R"] = data_mode_se["srgb_img"]
                data_pred_dicts["SR"] = (data_mode_sr, pred_mode_sr)

            # backward
            self.train_manager.optimizer.zero_grad()
            loss.backward()
            self.train_manager.optimizer.step()
            out = {
                "batch_size": data["srgb_img"].size(0),
                "total_loss": loss.item(),
                "loss_re": loss_mode_re.item() if train_re else 0.0,
                "loss_rr": loss_mode_rr.item() if train_rr else 0.0,
                "loss_se": loss_mode_se.item() if train_se else 0.0,
                "loss_sr": loss_mode_sr.item() if train_sr else 0.0,
                "data_pred_dicts": data_pred_dicts,
                "dataset": dataset
            }
            return out
        else:
            raise Exception(f"Can not train on dataset {dataset}!")

    def train_one_reflectance_estimation_step(self, dataset, data):
        data_mode_re = data.copy()
        pred_mode_re = self.model(data_mode_re["srgb_img"], mode="RE")
        if dataset in [C.CGIntrinsics, C.MPI_Sintel, C.Hypersim, C.MIT_Intrinsic]:
            t = "dense"
        elif dataset in [C.IIW]:
            t = "ordinal"
        else:
            assert False, f"Not support training re on {dataset}!"
        loss_mode_re = self.train_manager.criterion["R"][t](pred_mode_re, data_mode_re)
        if dataset == C.IIW:
            data_mode_re["gt_R"] = torch.zeros_like(data_mode_re["srgb_img"])
        return data_mode_re, pred_mode_re, loss_mode_re

    def save_model(self, out_dir, epoch, save_per_epoch, curr_val_result, best_val_result, cfg: CN):
        if (save_per_epoch > 0) and (epoch % save_per_epoch == 0):
            model_loader.save_model(self.model,
                                    out_dir, f"IID_{epoch:03d}",
                                    best_val_result.epoch == epoch,
                                    other_info={
                                        **curr_val_result.__dict__,
                                        "cfg": cfg,
                                    },
                                    model_only=False
                                    )
        else:
            if best_val_result.epoch == epoch:
                model_loader.save_model(self.model,
                                        os.path.join(cfg.DIR, "train/"), f"best_IID_{epoch:03d}",
                                        best_val_result.epoch == epoch,
                                        other_info={
                                            **best_val_result.__dict__,
                                            "cfg": cfg,
                                        },
                                        model_only=False
                                        )

    def visualize_data_pred(self, data_pred_dicts, batch_size):
        out = []
        for v_idx in range(0, min(2, batch_size)):
            m_grids = []
            for _, item in data_pred_dicts.items():
                data, pred = item  # data and pred from one trainable branch
                input_srgb, gt_R, pred_R = data["srgb_img"], data.get("gt_R", None), pred.get("pred_R", None)
                gt_S, pred_S = data.get("gt_S", None), pred.get("pred_S", None)
                dev = pred_R.device
                s_grid = []
                s_grid.append(input_srgb[v_idx].to(dev))
                if gt_R is not None:
                    # vis_gt_R = image_util.adjust_image_for_display(gt_R[v_idx].to(dev),
                    #                                            rescale=False, trans2srgb=False)
                    vis_gt_R = gt_R[v_idx].to(dev)
                    s_grid.append(vis_gt_R)
                if pred_R is not None:
                    # vis_pred_R = image_util.adjust_image_for_display(pred_R[v_idx].to(dev), rescale=False, trans2srgb=False)
                    vis_pred_R = pred_R[v_idx].to(dev)
                    s_grid.append(vis_pred_R)
                if gt_S is not None:
                    vis_gt_S = image_util.adjust_image_for_display(gt_S[v_idx].to(dev), rescale=False, trans2srgb=True)
                    s_grid.append(vis_gt_S)
                if pred_S is not None:
                    vis_pred_S = image_util.adjust_image_for_display(pred_S[v_idx].to(dev), rescale=False, trans2srgb=True)
                    s_grid.append(vis_pred_S)
                s_grid = torch.cat(s_grid, 2)
                # s_grid = torch.cat((input_srgb[v_idx].to(dev), vis_gt_R, vis_pred_R), 2)
                m_grids.append(s_grid)
            grid = torch.cat(m_grids, 1)
            # grid = F.interpolate(grid.unsqueeze(0), scale_factor=[0.5, 0.5], mode='area',recompute_scale_factor=False)[0]
            out.append((data["index"][v_idx].item(), grid))
        return out
