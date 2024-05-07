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
import argparse
from datetime import datetime

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode as CN

from configs.config import get_cfg_defaults
from utils import settings, logger
from utils.joint_data_loader import JointDataLoader, merge_datasets
from modeling import get_models
from solver.trainer import Trainer
from solver.train_manager import TrainingConfigManager
from solver import loss
from dataset import cgintrinsics_dataset, iiw_dataset, mpi_dataset
from dataset import mit_intrinsic_dataset
import constants as C


def make_dataset_cgintrinsics(cfg: CN, log, worker_init_fn=None, split=None, require_linear_input=False,
                              return_dataloader=True):
    if split is None:
        split = cfg.TRAIN.split
    assert split in ["train", "full"]
    dataset_cgi_train = cgintrinsics_dataset.CGIntrinsicsDataset(cfg.DATASET.CGIntrinsics_dir, split,
                                                                 (cfg.MODEL.input_img_size, cfg.MODEL.input_img_size),
                                                                 require_linear_input=require_linear_input)
    if cfg.MODE == "debug":
        print("============ Debug Mode ==============")
        dataset_cgi_train = torch.utils.data.Subset(dataset_cgi_train, range(0, 40))
        dataloader_cgi_train = torch.utils.data.DataLoader(dataset_cgi_train,
                                                           shuffle=False, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_cgi,
                                                           num_workers=cfg.TRAIN.workers_cgi,
                                                           worker_init_fn=worker_init_fn)
    else:
        dataloader_cgi_train = torch.utils.data.DataLoader(dataset_cgi_train,
                                                           shuffle=True, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_cgi,
                                                           num_workers=cfg.TRAIN.workers_cgi,
                                                           worker_init_fn=worker_init_fn)
    log.info(f"\nDataset:\n\tCGIntrinsics:"
             f"\t\t\ttrain: size: {len(dataset_cgi_train)}, batch_size: {cfg.TRAIN.batch_size_cgi}\n")
    # train_list_render_dir = os.path.join(cfg.DATASET.CGIntrinsics_dir, "intrinsics_final/render_list/")
    # dataset_render = Render_ImageFolder(root=cfg.DATASET.root_dir, list_dir=train_list_render_dir,
    #                                     transform=None)
    # data_loader_render = torch.utils.data.DataLoader(dataset_render, shuffle=True,
    #                                                  batch_size=cfg.TRAIN.batch_size_render,
    #                                                  num_workers=cfg.TRAIN.workers_render)
    # log.info(f"\n\tCGIntrinsics (render) _ train:\n"
    #          f"\t\tsize: {len(dataset_render)}, batch_size: {cfg.TRAIN.batch_size_render}")
    if return_dataloader:
        return len(dataset_cgi_train), JointDataLoader([dataloader_cgi_train], [C.CGIntrinsics]), cfg.TRAIN.batch_size_cgi
    else:
        return len(dataset_cgi_train), dataset_cgi_train, cfg.TRAIN.batch_size_cgi


def make_dataset_mpi_sintel(cfg: CN, log, worker_init_fn=None):
    assert cfg.TRAIN.split in ["image_front", "image_reverse", "scene_front", "scene_reverse"]
    # dataset CGI
    num_cgi, joint_loader_cgi, _ = make_dataset_cgintrinsics(cfg, log, worker_init_fn, "train")
    # dataset MPI Sintel
    dataset_mpi_train = mpi_dataset.MPIDataset(cfg.DATASET.MPI_Sintel_dir, cfg.TRAIN.split, "train",
                                               img_size=(cfg.MODEL.input_img_size, cfg.MODEL.input_img_size))
    if cfg.MODE == "debug":
        print("============ Debug Mode ==============")
        dataloader_mpi_train = torch.utils.data.DataLoader(dataset_mpi_train,
                                                           shuffle=False, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_cgi,
                                                           num_workers=cfg.TRAIN.workers_cgi,
                                                           worker_init_fn=worker_init_fn)
    else:
        dataloader_mpi_train = torch.utils.data.DataLoader(dataset_mpi_train,
                                                           shuffle=True, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_cgi,
                                                           num_workers=cfg.TRAIN.workers_cgi,
                                                           worker_init_fn=worker_init_fn)
    log.info(f"\nDataset:\n\tMPI_sintel({cfg.TRAIN.split}):"
             f"\t\ttrain: size: {len(dataset_mpi_train)}, batch_size: {cfg.TRAIN.batch_size_cgi}\n")
    # return len(dataset_mpi_train), JointDataLoader([dataloader_mpi_train], "MPI"), cfg.TRAIN.batch_size_cgi
    return len(dataset_mpi_train)+num_cgi, joint_loader_cgi.append(dataloader_mpi_train, C.MPI_Sintel), cfg.TRAIN.batch_size_cgi


def make_dataset_mit_intrinsic(cfg: CN, log, worker_init_fn=None):
    split = cfg.TRAIN.split
    assert split in ["train"]
    # dataset CGI
    _, dataset_cgi_train, _ = make_dataset_cgintrinsics(cfg, log, worker_init_fn, "train",
                                                        require_linear_input=True,
                                                        return_dataloader=False)
    # dataset MIT Intrinsic
    dataset_mit_train = mit_intrinsic_dataset.MITIntrinsicDataset(cfg.DATASET.MIT_Intrinsic_dir, split,
                                                                  img_size=(cfg.MODEL.input_img_size, cfg.MODEL.input_img_size))
    # merge datasets
    dataset_mix_train = merge_datasets(dataset_cgi_train, dataset_mit_train, balanced=True)
    if cfg.MODE == "debug":
        print("============ Debug Mode ==============")
        dataloader_train = torch.utils.data.DataLoader(dataset_mix_train,
                                                       batch_size=cfg.TRAIN.batch_size_cgi,
                                                       num_workers=cfg.TRAIN.workers_cgi,
                                                       worker_init_fn=worker_init_fn)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_mix_train,
                                                       shuffle=True, drop_last=True,
                                                       batch_size=cfg.TRAIN.batch_size_cgi,
                                                       num_workers=cfg.TRAIN.workers_cgi,
                                                       worker_init_fn=worker_init_fn)
    log.info(f"\nDataset:\n\tMIT_Intrinsic({cfg.TRAIN.split}):"
             f"\t\ttrain: size: {len(dataset_mit_train)}, batch_size: {cfg.TRAIN.batch_size_cgi}\n")
    log.info(f"\nDataset:\n\tCGIntrinsics + MIT_Intrinsic:"
             f"\t\ttrain: size: {len(dataset_mix_train)}, batch_size: {cfg.TRAIN.batch_size_cgi}\n")
    # return len(dataset_mit_train)+num_real, joint_loader_real.append(dataloader_mit_train, C.MIT_Intrinsic), \
    #        cfg.TRAIN.batch_size_cgi
    return len(dataset_mix_train), JointDataLoader([dataloader_train], [C.mix_CGI_MIT]), \
           cfg.TRAIN.batch_size_cgi


def make_dataset_iiw(cfg: CN, log, worker_init_fn=None):
    # dataset CGI
    num_cgi, joint_loader_cgi, _ = make_dataset_cgintrinsics(cfg, log, worker_init_fn, "train")
    # dataset IIW
    dataset_iiw_train = iiw_dataset.IIWDataset(cfg.DATASET.IIW_dir, -1, "train",
                                               img_size=(cfg.MODEL.input_img_size, cfg.MODEL.input_img_size),
                                               pseudo_gt_dir=cfg.DATASET.IIW_pred_by_NIID_Net)
    if cfg.MODE == "debug":
        print("============ Debug Mode ==============")
        dataset_iiw_train = torch.utils.data.Subset(dataset_iiw_train, range(0, 40))
        dataloader_iiw_train = torch.utils.data.DataLoader(dataset_iiw_train,
                                                           shuffle=False, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_iiw,
                                                           num_workers=cfg.TRAIN.workers_iiw,
                                                           collate_fn=iiw_dataset.custom_collate,
                                                           worker_init_fn=worker_init_fn)
    else:
        dataloader_iiw_train = torch.utils.data.DataLoader(dataset_iiw_train,
                                                           shuffle=True, drop_last=True,
                                                           batch_size=cfg.TRAIN.batch_size_iiw,
                                                           num_workers=cfg.TRAIN.workers_iiw,
                                                           collate_fn=iiw_dataset.custom_collate,
                                                           worker_init_fn=worker_init_fn)
    log.info(f"\nDataset:\n\tIIW:"
             f"\t\t\ttrain: size: {len(dataset_iiw_train)}, batch_size: {cfg.TRAIN.batch_size_iiw}\n")
    return len(dataset_iiw_train)+num_cgi, joint_loader_cgi.append(dataloader_iiw_train, C.IIW), cfg.TRAIN.batch_size_iiw


def make_dataset_hypersim(cfg: CN, log, worker_init_fn=None):
    assert cfg.TRAIN.split in ["train"]
    dataset_hypersim_train = hypersim_dataset.HypersimDataset(cfg.DATASET.Hypersim_dir, cfg.TRAIN.split)
    if cfg.MODE == "debug":
        print("============ Debug Mode ==============")
        dataset_hypersim_train = torch.utils.data.Subset(dataset_hypersim_train,
                                                         range(0, len(dataset_hypersim_train), len(dataset_hypersim_train)//40))
        dataloader_hypersim_train = torch.utils.data.DataLoader(dataset_hypersim_train,
                                                                shuffle=False, drop_last=True,
                                                                batch_size=cfg.TRAIN.batch_size_cgi,
                                                                num_workers=cfg.TRAIN.workers_cgi,
                                                                worker_init_fn=worker_init_fn)
    else:
        dataloader_hypersim_train = torch.utils.data.DataLoader(dataset_hypersim_train,
                                                                shuffle=True, drop_last=True,
                                                                batch_size=cfg.TRAIN.batch_size_cgi,
                                                                num_workers=cfg.TRAIN.workers_cgi,
                                                                worker_init_fn=worker_init_fn)
    log.info(f"\nDataset:\n\tHypersim:"
             f"\t\ttrain: size: {len(dataset_hypersim_train)}, batch_size: {cfg.TRAIN.batch_size_cgi}\n")
    return len(dataset_hypersim_train), JointDataLoader([dataloader_hypersim_train], [C.Hypersim]), cfg.TRAIN.batch_size_cgi


def train(cfg: CN):
    # Setting
    if not os.path.exists(cfg.DIR):
        os.makedirs(cfg.DIR)
    deterministic = cfg.TRAIN.seed != -1
    settings.set_(with_random=not deterministic, SEED=cfg.TRAIN.seed, deterministic=deterministic)

    # Logger
    log = logger.get_logger(os.path.join(cfg.DIR, "log.txt"), 1)
    writer_text = SummaryWriter(log_dir=os.path.join(cfg.DIR, "runs/"))
    writer_img = SummaryWriter(log_dir=os.path.join(cfg.DIR, "runs/"))
    writers = {
        "text": writer_text,
        "image": writer_img
    }

    # Model
    model = get_models.get(cfg)

    # Data loaders
    make_datasets = {
        C.CGIntrinsics: make_dataset_cgintrinsics,
        C.IIW: make_dataset_iiw,  # CGI + IIW
        C.Hypersim: make_dataset_hypersim,
        # C.BigTime: make_dataset_bigtime,
        C.MPI_Sintel: make_dataset_mpi_sintel,  # [CGI] + MPI
        C.MIT_Intrinsic: make_dataset_mit_intrinsic,  # CGI + MIT
    }
    assert cfg.TRAIN.dataset in make_datasets.keys(), f"Undefined dataset: {cfg.TRAIN.dataset}"
    worker_init_fn = settings.seed_worker if deterministic else None
    num_train, train_dataloaders, train_batch_size = make_datasets[cfg.TRAIN.dataset](cfg, log, worker_init_fn)

    # Criterion (training loss)
    p_dense = cfg.TRAIN.CRITERIA.DENSE
    p_ordinal = cfg.TRAIN.CRITERIA.ORDINAL
    criterion = {"R": {}, "S": {}}
    criterion["R"]["dense"] = \
        loss.DenseIntrinsicCriterion(model.color_rep, "R",
                                     p_dense.suppress_c, p_dense.suppress_i,
                                     p_dense.w_c, p_dense.w_i,
                                     p_dense.w_dense_value,
                                     p_dense.w_dense_grad,
                                     p_dense.w_dense_dssim)
    criterion["R"]["ordinal"] = \
        loss.IIWCriterion(w=p_ordinal.w_iiw, w_ineq=p_ordinal.w_ineq,
                          margin_eq=p_ordinal.margin_eq,
                          margin_ineq=p_ordinal.margin_ineq)
    criterion["S"]["dense"] = \
        loss.DenseIntrinsicCriterion(model.color_rep, "S",
                                     p_dense.suppress_c, p_dense.suppress_i,
                                     p_dense.w_c, p_dense.w_i,
                                     p_dense.w_dense_value,
                                     p_dense.w_dense_grad,
                                     p_dense.w_dense_dssim)

    # Train
    train_manager = TrainingConfigManager(cfg, criterion, train_dataloaders)
    trainer = Trainer()
    trainer.train(cfg, model, train_manager, writers, log)
    for k, w in writers.items():
        w.close()


if __name__ == '__main__':
    # update configuration
    opts = []
    # torch.set_num_threads(1)
    # opts = ["MODE", "debug",
    #         "MODEL.arch", "baseline_input",
    #         'MODEL.use_gpu', False,
    #         "TRAIN.save_per_epoch", 0,
    #         "TRAIN.dataset", "CGI",
    #         'TRAIN.batch_size_cgi', 2,
    #         'TRAIN.workers_cgi', 2,
    #         'TRAIN.batch_size_bigtime', 2,
    #         'VAL.batch_size_iiw', 8,
    #         'VAL.workers_iiw', 2,
    #         'TRAIN.epoch_iters', 5,
    #         'TRAIN.num_epoch', 20,
    #         "TRAIN.disp_iters", 3,
    #         "TRAIN.CRITERIA.type", "absolute",
    #         "TRAIN.CRITERIA.update_gt", False]  # options for code testing

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/crefnet_R_cgi.yaml",
        metavar="FILE",
        help="path to config file",
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
    cfg.merge_from_list(["DIR", os.path.join(cfg.DIR, datetime.now().strftime('%b%d_%H-%M-%S'))])
    cfg.freeze()  # freeze configuration

    # train the model
    train(cfg)
