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
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from yacs.config import CfgNode as CN
# from torchvision import io
import numpy as np

from utils import settings, model_loader, image_util
from configs.config import get_cfg_defaults
from modeling import get_models
from dataset import imagefolder_dataset
from solver.metrics_intrinsic_images import scale_matching


def infer_images(cfg: CN, img_dataset: torch.utils.data.Dataset,
                 out_dir: str, data_mode: str,
                 resize_input_res: tuple,
                 output_original_size: bool,
                 gamma_correct_input: bool,
                 gamma_correct_output: bool,
                 vis_wo_mask: bool):
    """
    Decompose images in the img_dir folder
    :param cfg: configuration
    :param img_dataset: dataset
    :param out_dir: output directory
    :param data_mode: mode for loading input images
    :param resize_input_res:
    :param output_original_size:
    :param gamma_correct_input:
    :param gamma_correct_output:
    :param vis_wo_mask: visualize without mask
    """
    print(f'Decompose {len(img_dataset)} images '
          f'from {img_dataset.data_dir} into {out_dir} with '
          f'\n\tmode {img_dataset.mode}, '
          f'\n\tresize_input {resize_input_res},'
          f'\n\toutput_original_size {output_original_size},'
          f'\n\tgamma_correct_input {gamma_correct_input},'
          f'\n\tgamma_correct_output {gamma_correct_output},'
          f'\n\tvis_wo_mask {vis_wo_mask}.')

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
    assert cfg.MODEL.checkpoint is not None, f"Should specify the path to the pretrained model."
    model_loader.load_model(model, cfg.MODEL.checkpoint, strict=True)
    model = model.to(device)
    model.freeze()

    # Decompose images
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    resize_input = resize_input_res[0] > 0 or resize_input_res[1] > 0
    if resize_input:
        print(f"     resize input to: {resize_input_res}.")
    else:
        print(f"     not resize input.")

    for idx in range(len(img_dataset)):
        # Read image
        data = img_dataset[idx]
        img_path = data["img_path"]
        img_name = data["img_name"]
        input_img = data["srgb_img"]  # [C, H, W]
        mask = data["mask"]
        # Predict
        input_img = input_img.unsqueeze(0).to(device)
        if gamma_correct_input:
            input_img = image_util.rgb_to_srgb(input_img)
        with torch.no_grad():
            pred = model(input_img, mode="IID", resize_input=resize_input,
                         min_img_size=resize_input_res[0], max_img_size=resize_input_res[1],
                         output_original_size=output_original_size)
        # Save results
        if vis_wo_mask:
            mask = torch.ones_like(pred["pred_R"][0])
        else:
            mask = mask.to(device)
            mask = F.interpolate(mask[None], size=pred["pred_R"].shape[2:],
                                 mode='bilinear', align_corners=False)[0]
            mask = (mask > 0.9).to(torch.float32)
        img_dicts = {
            "r": pred["pred_R"][0]*mask,
            "s": pred["pred_S"][0]*mask,
        }
        # rs = [input_img[0]]
        rs = []
        for k, img in img_dicts.items():
            # raw
            img_np = img.permute(1, 2, 0).cpu().numpy()
            np.save(os.path.join(out_dir, f"{img_name}_{k}.npy"), img_np)
            # tonemapping
            rescale = True
            if data_mode == "MIT_test":
                rescale = False
            elif img.size(1) * img.size(2) > 4000000:
                img = img / img.max().clamp(min=1e-6)
                rescale = False
            # if k == "s":
            #     if data_mode == "MIT_test":
            #         # img *= scale_matching(img[None], data["gt_S"].to(device)[None],
            #         #                       mask[None]).item()
            #         rescale = False
            #     elif img.size(1) * img.size(2) > 4000000:
            #         # rescale and avoid memory error
            #         img = img / img.max().clamp(min=1e-6)
            #         rescale = False
            # elif k == "r":
            #     if mode == "MIT_test":
            #         # img *= scale_matching(img[None], data["gt_R"].to(device)[None],
            #         #                       mask[None]).item()
            #         pass
            #     rescale = False
            # else:
            #     raise Exception(f"Unknown key {k}!")
            img = image_util.adjust_image_for_display(img, rescale=rescale, trans2srgb=gamma_correct_output,
                                                      src_percentile=0.99, dst_value=0.80)
            rs.append(img)
            save_image(img, os.path.join(out_dir, f"{img_name}_{k}.png"))
        save_image(rs, os.path.join(out_dir, f"{img_name}_result.jpg"))
        print(f"\t Decompose {idx+1}/{len(img_dataset)} {img_path} to {os.path.join(out_dir, f'{img_name}_result.jpg')} ......")
    return


if __name__ == '__main__':
    # update configuration
    opts = ["MODEL.checkpoint",
            None]
    # torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="configs/crefnet.yaml",
        metavar="FILE",
        help="Path to config file",
        type=str,
    )
    parser.add_argument(
        "--img-dir",
        default="examples/",
        metavar="FILE",
        help="Path to input images",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        default="",
        metavar="FILE",
        help="Path to output images",
        type=str,
    )
    parser.add_argument(
        "--loading_mode",
        default="none",
        help="Mode for loading input images. Options: none, with_mask, MIT_test",
        type=str,
    )
    parser.add_argument(
        "--gamma_correct_input",
        action="store_true",
        default=False,
        help="Whether to apply gamma correction to input images (new input = input ** (1.0/2.2))",
    )
    parser.add_argument(
        "--gamma_correct_output",
        action="store_true",
        help="Whether to apply gamma correction to output IID images (new out = out ** (1.0/2.2))",
    )
    parser.add_argument(
        "--min_input_dim",
        type=int,
        default=-1,
        help="Minimum input dimension for resizing the input data. The default value is -1.",
    )
    parser.add_argument(
        "--max_input_dim",
        type=int,
        default=-1,
        help="Maximum input dimension for resizing the input data. The default value is -1.",
    )
    parser.add_argument("--output_original_size", action="store_true", default=False,
                        help="Whether to output the original size.")
    parser.add_argument(
        "--vis_wo_mask", action="store_true",
        default=False,
        help="Whether to visualize predictions use mask."
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

    # dataloader: list all image files in the folder, filtered by mode
    img_dataset = imagefolder_dataset.ImageFolderLoader(args.img_dir, args.loading_mode)

    # predict
    out_dir = args.out_dir if args.out_dir != "" else os.path.join(args.img_dir, "out")
    infer_images(cfg, img_dataset, out_dir, args.loading_mode,
                 (args.min_input_dim, args.max_input_dim),
                 args.output_original_size,
                 args.gamma_correct_input,
                 args.gamma_correct_output,
                 args.vis_wo_mask)
