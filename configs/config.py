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


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "./experiments"
_C.MODE = "release"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dir = "./data/"
_C.DATASET.CGIntrinsics_dir = "./data/CGIntrinsics/intrinsics_final/"
_C.DATASET.IIW_dir = "./data/CGIntrinsics/IIW/"
_C.DATASET.IIW_pred_by_NIID_Net = None
_C.DATASET.Hypersim_dir = "./data/Hypersim/"
_C.DATASET.BigTime_v1_dir = "./data/BigTime_v1_resized"  # './data/phoenix/S6/zl548/AMOS/BigTime_v1/'
_C.DATASET.MPI_Sintel_dir = "./data/MPI_Sintel_IID"
_C.DATASET.SAW_dir = "./data/CGIntrinsics/SAW"
_C.DATASET.MIT_Intrinsic_dir = "./data/MIT-intrinsic"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.use_gpu = True
_C.MODEL.checkpoint = None
_C.MODEL.input_img_size = None
_C.MODEL.color_rep = "rgI"  # color space
_C.MODEL.ARCH = CN()
_C.MODEL.ARCH.name = "crefnet"  # architecture
_C.MODEL.ARCH.r_chs = 3  # number of R channels
_C.MODEL.ARCH.s_chs = -1  # number of S channels
_C.MODEL.ARCH.num_swin_groups = 8  # number of swin groups
_C.MODEL.ARCH.depth_per_swin_group = 4  # number of swin blocks per group

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.use_checkpoint_tool = True  # apply torch.utils.checkpoint on Swin to save GPU memory
_C.TRAIN.pretrained = None  # load pretrained model from this checkpoint file
_C.TRAIN.save_per_epoch = 0  # if <= 0, save a checkpoint file only when the model performance improves
_C.TRAIN.seed = 999  # manual seed
_C.TRAIN.dataset = "CGI"  # "CGI", "BigTime", "IIW"
_C.TRAIN.split = "train"  # also "full" for the CGI dataset
_C.TRAIN.batch_size_cgi = 8
_C.TRAIN.workers_cgi = 8
_C.TRAIN.batch_size_iiw = 8
_C.TRAIN.workers_iiw = 8
_C.TRAIN.batch_size_render = 1
_C.TRAIN.workers_render = 1
_C.TRAIN.batch_size_bigtime = 8
_C.TRAIN.workers_bigtime = 2
_C.TRAIN.num_epoch = 100
_C.TRAIN.epoch_iters = 2000
_C.TRAIN.disp_iters = 60  # frequency to display
_C.TRAIN.visualize = False
# Optimizer
_C.TRAIN.optim = "AdamW"
_C.TRAIN.lr = 2e-4
_C.TRAIN.weight_decay = 1e-4
# Scheduler
# _C.TRAIN.scheduler = "plateau"
# _C.TRAIN.schedular_mode = "max"
# _C.TRAIN.lr_decay = 0.5
# _C.TRAIN.sd_patience = 5
# _C.TRAIN.min_lr = 1e-6
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.name = "StepLR"  # None, "StepLR"
# _C.TRAIN.LR_SCHEDULER.delay = 0
_C.TRAIN.LR_SCHEDULER.decay_epoch = 15
# _C.TRAIN.LR_SCHEDULER.T_mult = 2
_C.TRAIN.LR_SCHEDULER.decay_rate = 0.25
_C.TRAIN.LR_SCHEDULER.warmup_epoch = 1
_C.TRAIN.LR_SCHEDULER.warmup_init_lr = 1e-6
# _C.TRAIN.LR_SCHEDULER.min_lr = 5e-5
# Strategy
_C.TRAIN.STRATEGY = CN()
_C.TRAIN.STRATEGY.type = "with_RR"   # "RE", "with_RR", "SE", "with_SR"
_C.TRAIN.STRATEGY.cosine_cycle = 15  # unused if type==None
_C.TRAIN.STRATEGY.rm_cgi_epoch = -1  # for training on the CGI+MPI
# Criterion
_C.TRAIN.CRITERIA = CN()
_C.TRAIN.CRITERIA.update_gt = False
# dense loss
_C.TRAIN.CRITERIA.DENSE = CN()
_C.TRAIN.CRITERIA.DENSE.type = "absolute"  # "si_cgi", "si_direct_intrinsics"
_C.TRAIN.CRITERIA.DENSE.suppress_c = 0.0  # suppression threshold for chromaticity
_C.TRAIN.CRITERIA.DENSE.suppress_i = 0.0  # suppression threshold for intensity
_C.TRAIN.CRITERIA.DENSE.w_c = 1.0
_C.TRAIN.CRITERIA.DENSE.w_i = 1.0
_C.TRAIN.CRITERIA.DENSE.w_dense_value = 1.0
_C.TRAIN.CRITERIA.DENSE.w_dense_grad = 40.0
_C.TRAIN.CRITERIA.DENSE.w_dense_dssim = 0.0
# ordinal loss
_C.TRAIN.CRITERIA.ORDINAL = CN()
_C.TRAIN.CRITERIA.ORDINAL.w_iiw = 1.0  # weight for IIW dataset
_C.TRAIN.CRITERIA.ORDINAL.w_ineq = 1.0  # weight for IIW inequality labels
_C.TRAIN.CRITERIA.ORDINAL.margin_eq = 0.0  # margin for hinge loss on the IIW dataset
_C.TRAIN.CRITERIA.ORDINAL.margin_ineq = 0.0  # margin for hinge loss on the IIW dataset
# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.batch_size_cgi = 16
_C.VAL.workers_cgi = 8
_C.VAL.batch_size_iiw = 1
_C.VAL.workers_iiw = 2
_C.VAL.batch_size_bigtime = 16
_C.VAL.workers_bigtime = 2
_C.VAL.visualize = True  # whether to output visualization during validation
_C.VAL.vis_per_iiw = 60
_C.VAL.vis_per_bigtime = 3
_C.VAL.vis_per_cgi = 240
_C.VAL.vis_per_mpi = 50
_C.VAL.vis_per_mit = 1
_C.VAL.whdr_eq_ratio = 0.1

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.dataset = "IIW"  # CGI or IIW or BigTime or MPI or SAW
_C.TEST.split = None
_C.TEST.batch_size_cgi = 16
_C.TEST.workers_cgi = 8
_C.TEST.batch_size_iiw = 1
_C.TEST.workers_iiw = 2
_C.TEST.batch_size_bigtime = 16
_C.TEST.workers_bigtime = 2
_C.TEST.visualize = True  # whether to output visualization during validation
_C.TEST.vis_per_iiw = 5
_C.TEST.vis_per_bigtime = 3
_C.TEST.vis_per_cgi = 240
_C.TEST.vis_per_mpi = 50
_C.TEST.vis_per_mit = 1
_C.TEST.seed = 999  # manual seed
_C.TEST.whdr_eq_ratio = 0.1
_C.TEST.dense_task = "R"  # dense evaluation on reflectance ("R") or shading ("S") estimation
_C.TEST.save_raw_pred = False

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
_C.VERSION = "v0.0"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern

  return _C.clone()
