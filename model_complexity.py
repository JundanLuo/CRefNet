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


import argparse
import time

import torch
from yacs.config import CfgNode as CN
import ptflops

from utils import settings
from configs.config import get_cfg_defaults
from modeling import get_models


def compute_model_complextiy(cfg: CN):
    """
    Compute the number of trainable parameters and GMac
    """

    # setting
    settings.set_(with_random=True, deterministic=False)

    # device
    if cfg.MODEL.use_gpu:
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available!")
        device = "cuda:0"
    else:
        device = "cpu"

    # model
    model = get_models.get(cfg)
    model = model.to(device)

    # set trainable parameters according to the training strategy
    model.configure_params_for_optimizer(cfg)
    model.unfreeze()

    # compute trainable parameters and GMac
    img_size = model.img_size
    macs, params = ptflops.get_model_complexity_info(
        model,  # the default forwarding mode for CRefNet is reflectance estimation.
                # you can change it in need
        (3, img_size, img_size),
        as_strings=True, print_per_layer_stat=False, verbose=True
    )
    print(f"With the training strategy {model.training_strategy} and input image size {(img_size, img_size)}")
    print('    {:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('    {:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(f'    => FLOPs: {macs:<8}, params: {params:<8}')

    # compute inference time
    model.freeze()
    input_data = torch.randn(1, 3, img_size, img_size).to(device)
    interval = 50
    end = time.time()
    for i in range(1000):
        with torch.no_grad():
            model(input_data, mode="RE", resize_input=False)
        if (i + 1) % interval == 0:
            print(f"Inference time of data {input_data.shape} per batch: "
                  f"{(time.time()-end)/interval}")
            end = time.time()
    return


if __name__ == '__main__':
    # update configuration
    opts = ["MODEL.checkpoint",
            None]
    # torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/refine_crefnet_R_cgi_iiw_448X.yaml",
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

    # model complexity
    compute_model_complextiy(cfg)
    