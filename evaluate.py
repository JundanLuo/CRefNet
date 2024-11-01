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


import math
import os
import argparse
from collections import namedtuple
from datetime import datetime
from utils import logger
import csv

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN
import numpy as np

from dataset.iiw_dataset import IIWDataset
from dataset import cgintrinsics_dataset, mpi_dataset, mit_intrinsic_dataset
from utils import image_util, settings, model_loader
from utils.average_meter import AverageMeter
from solver import metrics_iiw, metrics_intrinsic_images, metrics_saw
from configs.config import get_cfg_defaults
from modeling import get_models
import constants as C


class WHDRAverageMeter(object):
    Result = namedtuple("Result", ["WHDR", "WHDR_eq", "WHDR_ineq"])

    def __init__(self, name: str):
        self.name = name
        self.whdr_meter = AverageMeter("WHDR")
        self.whdr_eq_meter = AverageMeter("WHDR_eq")
        self.whdr_ineq_meter = AverageMeter("WHDR_ineq")

    def update(self, whdr, whdr_eq, whdr_ineq, count, count_eq, count_ineq):
        self.whdr_meter.update(whdr, count)
        self.whdr_eq_meter.update(whdr_eq, count_eq)
        self.whdr_ineq_meter.update(whdr_ineq, count_ineq)

    def get_results(self):
        return self.Result(WHDR=self.whdr_meter.avg, WHDR_eq=self.whdr_eq_meter.avg, WHDR_ineq=self.whdr_ineq_meter.avg)

    def __str__(self):
        return f"WHDR {self.whdr_meter.avg: .5f}, "\
            f"WHDR_eq {self.whdr_eq_meter.avg: .5f}, "\
            f"WHDR_ineq {self.whdr_ineq_meter.avg: .5f}"


def validate_iiw(model, device, cfg: CN, mode: str,
                 display_process=True,
                 writer=None,
                 visualize_dir=None,
                 label='val_iiw',
                 csv_writer=None,
                 raw_pred_dir=None):
    """Evaluate reflectance estimation of the model

    :param model:
    :param device:
        the device for input data
    :param cfg:
        configuration
    :param mode:
        "test": test split
        "val": val split
    :param display_process:
        print the evaluation process or not
    :param writer:
        tensorboard writer or None
    :param visualize_dir:
        directory for visualized results
    :param label:
        prefix of the output file name
    :param csv_writer:
        a CSV file that records the WHDR scores of each image
    :param raw_pred_dir:
        save raw predictions into .npy files
    """

    print(f"=========================== Evaluation ON IIW: {mode}============================")
    if mode == "test":  # test
        batch_size_iiw = cfg.TEST.batch_size_iiw
        num_workers_iiw = cfg.TEST.workers_iiw
        assert batch_size_iiw == 1, f"Only supports evaluation on IIW {mode} set with batch_size=1"
        vis_per_images = cfg.TEST.vis_per_iiw
        input_img_size = None  # not resize input image
        whdr_eq_ratio = cfg.TEST.whdr_eq_ratio
    elif mode == "val":  # val
        batch_size_iiw = cfg.VAL.batch_size_iiw
        num_workers_iiw = cfg.VAL.workers_iiw
        assert batch_size_iiw == 1, f"Only supports evaluation on IIW {mode} set with batch_size=1"
        vis_per_images = cfg.VAL.vis_per_iiw
        input_img_size = None  # not resize input image
        whdr_eq_ratio = cfg.VAL.whdr_eq_ratio
    else:
        raise Exception(f"Erroneous mode for validate iiw: {mode}")

    print(f'batch size for iiw: {batch_size_iiw}')

    if raw_pred_dir is not None:
        os.makedirs(raw_pred_dir, exist_ok=True)
    if csv_writer is not None:
        assert batch_size_iiw == 1
        header = ["num", "index", "whdr_overall", "whdr_eq", "eq_ratio"]
        csv_writer.writerow(header)
    sum_num_imgs = 0
    whdr_rgb_meter = WHDRAverageMeter("whdr_rgb")
    whdr_srgb_meter = WHDRAverageMeter("whdr_srgb")
    for j in range(0, 3):  # for 3 different orientation
        # data loader
        dataset_iiw = IIWDataset(cfg.DATASET.IIW_dir, j, mode, img_size=input_img_size)
        data_loader_iiw = torch.utils.data.DataLoader(dataset_iiw,
                                                      batch_size=batch_size_iiw,
                                                      num_workers=num_workers_iiw,
                                                      shuffle=False,
                                                      drop_last=False)
        sum_num_imgs += len(dataset_iiw)

        # validate
        model.freeze()  # freeze model
        writing_index = 0
        for i, data_iiw in enumerate(data_loader_iiw):
            # predict
            input_srgb = data_iiw["srgb_img"].to(torch.float32).to(device)
            targets = data_iiw["targets"]
            with torch.no_grad():
                pred = model(input_srgb, mode="IID")
                pred_R = pred["pred_R"].detach()
                pred_S = pred["pred_S"].detach()
                # direct_linear_out_R = pred["direct_linear_out_R"].detach()

            # compute WHDR
            (total_whdr, count), (total_whdr_eq, count_eq), (total_whdr_ineq, count_ineq) = \
                metrics_iiw.evaluate_WHDR(pred_R, targets, whdr_eq_ratio)
            whdr_rgb_meter.update(total_whdr/max(count, 1e-6), total_whdr_eq/max(count_eq, 1e-6), total_whdr_ineq/max(count_ineq, 1e-6),
                                  count, count_eq, count_ineq)

            (total_whdr, count), (total_whdr_eq, count_eq), (total_whdr_ineq, count_ineq) = \
                metrics_iiw.evaluate_WHDR(image_util.adjust_image_for_display(pred_R, False, True, clip=True), targets,
                                          whdr_eq_ratio)
            whdr_srgb_meter.update(total_whdr/max(count, 1e-6), total_whdr_eq/max(count_eq, 1e-6), total_whdr_ineq/max(count_ineq, 1e-6),
                                  count, count_eq, count_ineq)

            # record and visualize
            if csv_writer is not None:
                csv_writer.writerow([f"{j}-{i}",
                                     data_iiw['img_name'][0],
                                     total_whdr/max(count, 1e-6),
                                     total_whdr_eq/max(count_eq, 1e-6),
                                     whdr_eq_ratio])
            if display_process:
                print(f"Evaluate {j}-{i} with WHDR eq_ratio {whdr_eq_ratio:.3f}: \n"
                      f"\tWHDR(rgb) {whdr_rgb_meter} \n"
                      f"\tWHDR(srgb) {whdr_srgb_meter}")
            if visualize_dir is not None:
                if i % math.ceil(vis_per_images // batch_size_iiw) == 0:
                    idx = 0
                    img_name = data_iiw['img_name'][idx]
                    vis_R = image_util.adjust_image_for_display(pred_R[idx], rescale=True, trans2srgb=False)
                    vis_S = image_util.adjust_image_for_display(pred_S[idx], rescale=True, trans2srgb=False,
                                                                src_percentile=0.95, dst_value=0.8)
                    # vis_direct_linear_out_R = image_util.adjust_image_for_display(direct_linear_out_R[idx],
                    #                                                               rescale=True, trans2srgb=False)
                    # vis_merge = torch.cat((input_srgb[idx].to(vis_R.device), vis_R, vis_direct_linear_out_R), 2)
                    vis_merge = torch.cat((input_srgb[idx].to(vis_R.device), vis_R, vis_S), 2)
                    image_util.save_srgb_image(vis_merge, visualize_dir,
                                               f"{label}_{j}-{data_iiw['index'][idx]}_{img_name}_result.jpg")
                    if mode == "test":
                        vis_dict = {
                            # "input": input_srgb[idx].to(vis_R.device),
                            "r": vis_R,
                            "s": vis_S
                        }
                        for key, item in vis_dict.items():
                            image_util.save_srgb_image(item, os.path.join(visualize_dir, "split"),
                                                       f"{img_name}_{key}.jpg")
            if writer is not None:
                if i % math.ceil(600 // batch_size_iiw) == 0:
                    idx = 0
                    vis_R = image_util.adjust_image_for_display(pred_R[idx], rescale=True, trans2srgb=False)
                    vis_grid = torch.cat((input_srgb[idx].to(vis_R.device), vis_R), 2)
                    writer.add_image(f"IIW/{label}", vis_grid, global_step=writing_index, dataformats="CHW")
                    writing_index += 1
            if raw_pred_dir is not None:
                for idx in range(pred_R.size(0)):
                    img_name = data_iiw['img_name'][idx]
                    pred_r_np = pred_R[idx].permute(1, 2, 0).cpu().numpy()
                    np.save(os.path.join(raw_pred_dir, f"{img_name}_r.npy"), pred_r_np)
                    pred_s_np = pred_S[idx].permute(1, 2, 0).cpu().numpy()
                    np.save(os.path.join(raw_pred_dir, f"{img_name}_s.npy"), pred_s_np)

    result_rgb = whdr_rgb_meter.get_results()
    result_srgb = whdr_srgb_meter.get_results()
    print(f"Evaluate {sum_num_imgs} in IIW dataset with WHDR eq_ratio {whdr_eq_ratio:.3f}:\n"
          f"WHDR(rgb) {result_rgb.WHDR:.5f}, WHDR(srgb) {result_srgb.WHDR:.5f}\n"
          f"WHDR_eq(rgb) {result_rgb.WHDR_eq:.5f}, WHDR_ineq(rgb) {result_rgb.WHDR_ineq:.5f}\n"
          f"WHDR_eq(srgb) {result_srgb.WHDR_eq:.5f}, WHDR_ineq(srgb) {result_srgb.WHDR_ineq:.5f}\n")
    return result_rgb, result_srgb


def validate_saw(model, device, cfg: CN, mode: str,
                 challenge_metric=True,
                 display_process=True,
                 writer=None,
                 visualize_dir=None,
                 label='val_saw'):
    """ Evaluate shading estimation of the model
    :param model:
    :param device:
        the device for input data
    :param cfg:
        configuration
    :param mode:
        "test": test split
        "val": val split
    :param challenge_metric:
        False: unweighted precision 0 (P(u))
        True : challenge precision 1 (P(c))
    :param display_process:
        print the evaluation process or not
    :param writer:
        tensorboard writer or None
    :param visualize_dir:
        directory for the PR_array and visualized results
    :param label:
        prefix of the name of the output file
    :return:
        AP result
    """
    print(f"============================ Validation ON SAW: {mode}============================")
    # parameters for SAW
    pixel_labels_dir = os.path.join(cfg.DATASET.SAW_dir, "saw_pixel_labels",
                                    "saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0")
    splits_dir = os.path.join(cfg.DATASET.SAW_dir, "saw_splits")
    img_dir = os.path.join(cfg.DATASET.SAW_dir, "saw_images_512")
    if mode == "test":
        dataset_split = "E"
        use_subset = False
        samples = 80
    elif mode == "val":
        dataset_split = "R"
        use_subset = True
        samples = 20
    else:
        assert False, f"Error mode: {mode}"
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    AP, plot_arr, sample_arr = metrics_saw.compute_pr(model, device,
                                                      pixel_labels_dir, splits_dir,
                                                      dataset_split, class_weights, bl_filter_size,
                                                      img_dir,
                                                      mode=1 if challenge_metric else 0,
                                                      display_process=display_process, samples=samples,
                                                      use_subset=use_subset)
    mm = plot_arr[:, 0] * plot_arr[:, 1]
    sm = plot_arr[:, 0] + plot_arr[:, 1]
    f1_score = 2 * (mm / sm)
    max_f1 = f1_score.max()
    
    # Save visualized results of samples and PR_array
    if visualize_dir is not None:
        for sample in sample_arr:
            img_name, pred_R, pred_S, input_srgb = sample["img_name"], sample["pred_R"][0], sample["pred_S"][0], \
                                                   sample["input_srgb"][0]
            vis_R = image_util.adjust_image_for_display(pred_R, rescale=True, trans2srgb=False)
            vis_S = image_util.adjust_image_for_display(pred_S, rescale=True, trans2srgb=False,
                                                        src_percentile=0.95, dst_value=0.8)
            input_srgb = input_srgb.to(vis_R.device)
            vis_merge = torch.cat((input_srgb, vis_R, vis_S), 2)
            image_util.save_srgb_image(vis_merge, visualize_dir,
                                       f"{label}_{img_name}_result.jpg")
            if mode == "test":
                vis_dict = {
                    "input": input_srgb,
                    "r": vis_R,
                    "s": vis_S
                }
                for key, item in vis_dict.items():
                    image_util.save_srgb_image(item, os.path.join(visualize_dir, "split"),
                                               f"{img_name}_{key}.jpg")
    print(f"Evaluate on SAW dataset:\n"
          f"AP {AP:.5f},    max f1: {max_f1:.5f}")
    return AP, max_f1, plot_arr


def test(cfg: CN):
    """ Evaluate intrinsic images on test sets
    """
    dataset = cfg.TEST.dataset

    # setting
    settings.set_(with_random=False, SEED=cfg.TEST.seed, deterministic=True)

    # device
    if cfg.MODEL.use_gpu:
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available!")
        device = "cuda:0"
    else:
        device = "cpu"

    # model
    model = get_models.get(cfg)
    if cfg.MODEL.checkpoint is not None:
        model_loader.load_model(model, cfg.MODEL.checkpoint, strict=True)
    model = model.to(device)

    # logger
    filename = os.path.basename(cfg.MODEL.checkpoint)[:-3] if cfg.MODEL.checkpoint is not None else ""
    test_out_dir = os.path.join(cfg.DIR, "test", datetime.today().strftime('%Y-%m-%d'), f"test_with_{filename}")
    os.makedirs(test_out_dir, exist_ok=True)
    log = logger.get_logger(os.path.join(test_out_dir, "log.txt"), 1, file_mode="a")
    log.info(f'Configuration:\n'
             f'{str(cfg)}')
    writer_img = SummaryWriter(log_dir=os.path.join(test_out_dir, "runs/"))

    # test
    model.freeze()
    visualize_dir = os.path.join(test_out_dir, dataset) if cfg.TEST.visualize else None
    raw_pred_dir = os.path.join(test_out_dir, f"raw_pred_{dataset}") if cfg.TEST.save_raw_pred else None
    if dataset == C.IIW:
        csv_file = open(os.path.join(test_out_dir, f"results_{dataset}.csv"), "w", encoding='UTF8', newline='')
        csv_writer = csv.writer(csv_file)
        result_rgb, result_srgb = validate_iiw(model, device,
                                               cfg, "test",
                                               True,
                                               writer=writer_img,
                                               visualize_dir=visualize_dir,
                                               label='test_iiw',
                                               csv_writer=csv_writer,
                                               raw_pred_dir=raw_pred_dir)
        print(f'Test {dataset}: WHDR(rgb) {result_rgb.WHDR:.5f}, WHDR(srgb) {result_srgb.WHDR:.5f}')
        log.info(f"Test {dataset}\n"
                 f"\t RGB space: {result_rgb}\n"
                 f"\t sRGB space: {result_srgb}\n\n")
        # close file
        csv_file.close()
    elif dataset == C.SAW:
        AP, max_f1, plot_arr = validate_saw(model, device,
                                    cfg, "test",
                                    True,
                                    True, writer=None,
                                    visualize_dir=visualize_dir,
                                    label="test_saw")
        print(f"Test {dataset} : AP(c) {AP:.5f}, max f1: {max_f1:.5f}")
        log.info(f"Test {dataset} : AP(c) {AP:.5f}, max f1: {max_f1:.5f}")
        # save PR_array
        file_path = os.path.join(test_out_dir, 'saw_plot_arr_c.npy')
        with open(file_path, 'wb') as f:
            np.save(f, plot_arr)
        print(f'save plot_arr{plot_arr.shape}: {file_path}')
    elif dataset in [C.CGIntrinsics, C.MPI_Sintel, C.MIT_Intrinsic]:
        split = cfg.TEST.split if dataset == C.MPI_Sintel else None
        si_MSE, si_LMSE, DSSIM = validate_dense(dataset, split,
                                                model, device,
                                                cfg, "test",
                                                val=cfg.TEST.dense_task,
                                                display_process=True,
                                                writers={
                                                  "text": writer_img,
                                                  "image": writer_img
                                                },
                                                visualize_dir=visualize_dir,
                                                label=f'test_{dataset}_{split}',
                                                raw_pred_dir=raw_pred_dir)
        print(f"Test {dataset}_split({split}): si_MSE {si_MSE:.6f}, si_LMSE {si_LMSE:.6f}, DSSIM {DSSIM:.6f}")
        log.info(f"Test {dataset}_split({split}): si_MSE {si_MSE:.6f}, si_LMSE {si_LMSE:.6f}, DSSIM {DSSIM:.6f}\n\n")
    elif dataset == "BigTime":
        from evaluate_image_sequence import validate_image_sequences
        validate_image_sequences(model, device,
                                 cfg,
                                 "BigTime_v1", "test",
                                 True,
                                 writers={"image": writer_img},
                                 visualize_dir=visualize_dir,
                                 label="test_bigtime")
    else:
        assert False, f"Not support dataset {dataset}!"


def validate_dense(dataset: str, split,
                   model, device, cfg: CN, mode: str,
                   val="R",
                   display_process=True,
                   writers=None,
                   visualize_dir=None,
                   epoch=-1,
                   label="val",
                   raw_pred_dir=None):
    print(f"=========================== Evaluation ON {dataset}: {val}, {mode}============================")
    assert val in ["R", "S"], f"Not support val on {val}!"
    if dataset == C.CGIntrinsics:
        # test/val dataset
        if mode == "test":  # test
            dataset_dense = cgintrinsics_dataset.CGIntrinsicsDataset(cfg.DATASET.CGIntrinsics_dir, mode, None)
            batch_size = cfg.TEST.batch_size_cgi
            num_workers = cfg.TEST.workers_cgi
            vis_per_img = cfg.TEST.vis_per_cgi
        elif mode == "val":  # val
            dataset_dense = cgintrinsics_dataset.CGIntrinsicsDataset(cfg.DATASET.CGIntrinsics_dir, mode, None)
            batch_size = cfg.VAL.batch_size_cgi
            num_workers = cfg.VAL.workers_cgi
            vis_per_img = cfg.VAL.vis_per_cgi
        else:
            raise Exception(f"Erroneous mode for validate cgi: {mode}")
        resize_input = True
    elif dataset == C.MPI_Sintel:
        assert mode in ["test", "val"]
        dataset_dense = mpi_dataset.MPIDataset(cfg.DATASET.MPI_Sintel_dir, split, mode, img_size=None)
        batch_size = cfg.TEST.batch_size_cgi
        num_workers = cfg.TEST.workers_cgi
        vis_per_img = cfg.TEST.vis_per_mpi
        resize_input = False
    elif dataset == C.MIT_Intrinsic:
        assert mode in ["test", "val"]
        dataset_dense = mit_intrinsic_dataset.MITIntrinsicDataset(cfg.DATASET.MIT_Intrinsic_dir, mode, img_size=None)
        batch_size = 1
        num_workers = 1
        vis_per_img = cfg.TEST.vis_per_mit
        resize_input = False
    else:
        assert False

    dataloader = torch.utils.data.DataLoader(dataset_dense,
                                             shuffle=False, drop_last=False,
                                             batch_size=batch_size,
                                             num_workers=num_workers)
    print(f"validate on {dataset}:\n"
          f"\tsize of dataset: {len(dataset_dense)}, batch_size: {batch_size}\n"
          )

    #
    if writers is not None:
        writer_text = writers["text"]
        writer_img = writers["image"]
    else:
        writer_text = writer_img = None
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        split_img_dir = os.path.join(visualize_dir, "split")
        if not os.path.exists(split_img_dir):
            os.makedirs(split_img_dir)
    if raw_pred_dir is not None and not os.path.exists(raw_pred_dir):
        os.makedirs(raw_pred_dir)

    # validate
    model.freeze()
    writing_index = 0
    meter = metrics_intrinsic_images.\
        SI_IntrinsicImageMetricsMeter(f"{val}_meter", True, mode == "test", True, mode)

    np_alpha = np.empty(0)
    for i, data in enumerate(dataloader):
        for k in ["srgb_img", "gt_R", "gt_S", "mask"]:
            data[k] = data[k].to(device, dtype=torch.float32).requires_grad_(False)
        assert data["srgb_img"].shape == data["gt_R"].shape ==\
               data["gt_S"].shape == data["mask"].shape, "Inconsistent shape"
        with torch.no_grad():
            pred = model(data["srgb_img"], mode="IID", resize_input=resize_input)
        meter.update(pred[f"pred_{val}"], data[f"gt_{val}"], data["mask"])

        # show alphas for sampled predictions
        # if writer_text is not None:
        #     alpha = metrics_intrinsic_images.scale_matching(pred_R, gt_R, mask)
        #     np_alpha = np.append(np_alpha, alpha.view(alpha.size(0)).cpu().numpy())
        #     if i % 10 == 0:
        #         idx = 0
        #         writer_text.add_scalar(f"{mode.title()}/{dataset}(index:{data['index'][idx]})", alpha[idx], global_step=epoch)

        # print evaluation results
        if display_process:
            rs = meter.get_results()
            print(f"Evaluate: {i}/{len(dataloader)}, "
                  f"si_MSE: {rs.si_MSE: .6f}, "
                  f"si_LMSE: {rs.si_LMSE: .6f}, "
                  f"DSSIM: {rs.DSSIM: .6f}")

        # visualize predicted images
        if (visualize_dir is not None and i % max(math.ceil(vis_per_img // batch_size), 1) == 0) or \
                (writers is not None and i % max(math.ceil(vis_per_img*3 // batch_size), 1) == 0):
            idx = 0
            # vis_pred_R = image_util.adjust_image_for_display(pred["pred_R"][idx], rescale=False, trans2srgb=False,)
            #                                                  # src_percentile=0.9999, dst_value=0.80)
            # vis_gt_R = image_util.adjust_image_for_display(data["gt_R"][idx], rescale=False,
            #                                                trans2srgb=False)
            vis_gt_R = data["gt_R"][idx]
            vis_gt_S = data["gt_S"][idx]
            mask = data["mask"][idx]
            vis_pred_R = pred["pred_R"][idx]
            vis_pred_R = image_util.adjust_image_for_display(vis_pred_R, rescale=True, trans2srgb=False,
                                                             src_percentile=0.99, dst_value=0.80)
            vis_pred_S = pred["pred_S"][idx]
            vis_pred_S = image_util.adjust_image_for_display(vis_pred_S, rescale=True, trans2srgb=False,
                                                             src_percentile=0.99, dst_value=0.80)
            if dataset == C.MIT_Intrinsic:
                vis_pred_R = vis_pred_R * mask
                vis_pred_S = vis_pred_S * mask
            vis_grid_1 = make_grid([data["srgb_img"][idx], vis_gt_R, vis_pred_R])
            vis_grid_2 = make_grid([mask, vis_gt_S, vis_pred_S])
            vis_grid = torch.cat([vis_grid_1, vis_grid_2], dim=1)
            vis_grid = F.interpolate(vis_grid.unsqueeze(0),
                                     scale_factor=[0.5, 0.5], mode='area', recompute_scale_factor=False)[0]
            if visualize_dir is not None and i % max(math.ceil(vis_per_img // batch_size), 1) == 0:
                save_image(vis_grid, os.path.join(visualize_dir, f"{label}_{data['img_name'][idx]}_result.png"))
                if mode == "test":
                    vis_dict = {
                        # "input": data["srgb_img"][idx].to(vis_pred_R.device),
                        # "gt_r": vis_gt_R,
                        # "gt_s": vis_gt_S,
                        "r": vis_pred_R,
                        "s": vis_pred_S,
                    }
                    for key, item in vis_dict.items():
                        save_image(item, os.path.join(split_img_dir,
                                                      f"{data['img_name'][idx]}_{key}.png"))
                        # pred_np = item.permute(1, 2, 0).cpu().numpy()
                        # np.save(os.path.join(visualize_dir, "split",
                        #                      f"{data['img_name'][idx]}_{key}.jpg"), pred_np)
            if writer_img is not None and i % max(math.ceil(vis_per_img*3 // batch_size), 1) == 0:
                writer_img.add_image(f"{dataset}/{label}", vis_grid, global_step=writing_index, dataformats="CHW")
                writing_index += 1
        if raw_pred_dir is not None:
            for idx in range(pred["pred_R"].size(0)):
                img_name = data['img_name'][idx]
                for k in ["R", "S"]:
                    pred_np = pred[f"pred_{k}"][idx].permute(1, 2, 0).cpu().numpy()
                    np.save(os.path.join(raw_pred_dir, f"{img_name}_{k.lower()}.npy"), pred_np)


    # show the histogram of alphas for all the predictions
    if writer_text is not None:
        hist_img = image_util.generate_histogram_image(np_alpha, f"{dataset}({mode})_alphas_for_pred", "alpha", "count",
                                                       200, [0, 20])
        writer_text.add_image(f'{mode.title()}/{dataset}_alphas', hist_img, global_step=epoch)

    # Result
    rs = meter.get_results()
    print(f"Evaluate {len(dataset_dense)} in {dataset}_{mode} set:\n"
          f"\tsi_MSE: {rs.si_MSE: .6f}, "
          f"\tsi_LMSE: {rs.si_LMSE: .6f}, "
          f"\tDSSIM: {rs.DSSIM: .6f}")
    return rs.si_MSE, rs.si_LMSE, rs.DSSIM


if __name__ == '__main__':
    # update configuration
    opts = ["MODEL.checkpoint",
            None]
    # torch.set_num_threads(1)
    # opts = [
    #     "DIR", "./experiments/test",
    #     'MODEL.use_gpu', False,
    #     "MODEL.arch", "baseline_constant",
    #     'TRAIN.batch_size_cgi', 4,
    #     'TRAIN.workers_cgi', 2,
    #     'TRAIN.workers_render', 1,
    #     'VAL.batch_size_iiw', 8,
    #     'VAL.workers_iiw', 2,
    #     'TRAIN.epoch_iters', 10,
    #     'TRAIN.num_epoch', 10,
    #     "TRAIN.disp_iters", 5,
    #     "MODEL.checkpoint", None,
    #     "TEST.dataset", "IIW"]  # options for code testing

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/crefnet.yaml",
        metavar="FILE",
        help="Path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("\nyaml file path: ", args.cfg)
    print("user defined options: ", opts + args.opts)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(opts + args.opts)
    cfg.freeze()  # freeze configuration
    print("\n", cfg, "\n")

    # test the model
    test(cfg)

