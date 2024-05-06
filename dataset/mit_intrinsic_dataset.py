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
import os.path
import time

import torch, torchvision
import torch.utils.data as data
import numpy as np
# from torchvision import io
import cv2
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F
from kornia.geometry import resize

from utils import image_util
from utils.image_util import get_scale_alpha, read_image
import constants as C


# MAX_BIT = 65025.0  # this is the maximum value of a mask in the mit intrinsic dataset


class MITIntrinsicDataset(data.Dataset):
    """
    Load MIT Intrinsic dataset
    """
    dataset_name = C.MIT_Intrinsic
    data_dir = "data"
    # train/test split by: Jon Barron and Jitendra Malik's SIRFS (Shape, Illumination, and Reflectance from Shading)
    # https://jonbarron.info/
    objects_dir = {
        "train": ['apple', 'box', 'cup1', 'dinosaur', 'frog1',
                  'panther', 'paper1', 'phone', 'squirrel', 'teabag2'],
        "test": ['cup2', 'deer', 'frog2', 'paper2', 'pear',
                 'potato', 'raccoon', 'sun', 'teabag1', 'turtle']
    }
    GAMMA = 1.0/2.2

    def __init__(self, root: str,
                 mode: str,
                 img_size=None
                 ) -> None:
        assert mode in ["train", "test", "val"]
        assert mode in ["train"] or img_size is None, f"Don't resize {mode} images."
        if mode == "val":
            mode = "test"
        self.mode = mode
        self.is_train = (self.mode in ["train"])
        self.img_size = img_size

        # check dataset path
        self.root = root
        self.data_dir = os.path.join(self.root, self.data_dir)
        self.data_list = self._get_data_list()

    def __len__(self):
        return len(self.data_list)

    def _get_data_list(self):
        obj_dirs = self.objects_dir[self.mode]
        obj_dirs = [os.path.join(self.data_dir, p) for p in obj_dirs]
        flag = self._check_exists([
            self.root,
            self.data_dir,
        ] + obj_dirs)
        if not flag:
            raise RuntimeError(f"{self.dataset_name} is not found or not complete "
                               f"in the path: {self.root}")
        # load list
        data_list = []
        if self.mode in ["train"]:
            for d in obj_dirs:
                data_list.append(os.path.join(d, "diffuse.png"))
                for idx in range(1, 11, 1):
                    data_list.append(os.path.join(d, f"light{idx:02d}.png"))
        elif self.mode in ["test"]:
            for d in obj_dirs:
                data_list.append(os.path.join(d, "diffuse.png"))
        else:
            assert False, f"Error mode {self.mode}"
        return data_list

    def _check_exists(self, paths) -> bool:
        flag = True
        for p in paths:
            flag = flag and os.path.exists(p)
        return flag

    def load_images(self, path: str, augment_data: bool):
        # load images
        obj_path, filename = os.path.split(path)
        _, obj = os.path.split(obj_path)

        rgb_img_path = path
        gt_R_path = os.path.join(obj_path, "reflectance.png")
        mask_path = os.path.join(obj_path, "mask.png")

        # Notice: MIT Intrinsic dataset is 16-bit png
        rgb_img = read_image(rgb_img_path, "tensor")
        gt_R = read_image(gt_R_path, "tensor")
        mask = read_image(mask_path, "tensor")[None, :, :].repeat(3, 1, 1)
        mask = (mask > 0.5).to(torch.float32)
        if filename.endswith("diffuse.png"):
            gt_S_path = os.path.join(obj_path, "shading.png")
            gt_S = read_image(gt_S_path, "tensor")[None, :, :].repeat(3, 1, 1)
        else:
            gt_S = rgb_img / gt_R.clamp(min=1e-6) * mask
            gt_S = gt_S.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        # print(f"srgb_img: {srgb_img.max()}")
        # print(f"gt_R: {gt_R.max()}")
        # print(f"gt_S: {gt_S.max()}")
        # print(f"mask: {mask.max()}")

        # data augmentation
        if augment_data:
            # color jitter and scale
            gt_R = torchvision.transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5)(gt_R) * mask
            alpha = get_scale_alpha(gt_R, mask, src_percentile=0.9999, dst_value=0.90).clamp(min=0.001, max=255).item()
            gt_R = (gt_R * alpha).clamp(min=0.0, max=1.0)
            gt_S = gt_S / alpha
            # reconstruct rgb image
            rgb_img = gt_R * gt_S * mask
            # random resized crop
            data_tuple = (rgb_img, gt_R, gt_S, mask)
            if self.img_size is not None:
                i, j, h, w = RandomResizedCrop.get_params(rgb_img, scale=[0.1, 1.0], ratio=[4. / 10., 10. / 4.])
                data_tuple = (F.crop(d, i, j, h, w) for d in data_tuple)
                data_tuple = (resize(d, size=self.img_size, interpolation="bilinear", align_corners=False, antialias=True)
                              for d in data_tuple)
            if torch.rand(1) < 0.5:
                data_tuple = (F.hflip(d) for d in data_tuple)
            if torch.rand(1) < 0.5:
                data_tuple = (F.vflip(d) for d in data_tuple)
            rgb_img, gt_R, gt_S, mask = data_tuple
            # mask = (mask > 0.99).to(torch.float32)
            # gt_S *= image_util.get_scale_alpha(gt_S, mask, 0.9, 0.8)
            # gt_S = gt_S.clamp(min=0.0, max=1.0)

        mask = (mask > 0.99).to(torch.float32)
        gt_R = gt_R * mask
        # gt_S = gt_S * mask
        gt_S[mask == 0] = 0.0
        srgb_img = rgb_img
        # srgb_img = image_util.rgb_to_srgb(rgb_img, self.GAMMA)
        return srgb_img, rgb_img, gt_R, gt_S, mask, f"{obj}_{filename[:-4]}"

    def __getitem__(self, index):
        srgb_img, rgb_img, gt_R, gt_S, mask, filename = self.load_images(self.data_list[index], self.is_train)
        return {"srgb_img": srgb_img,
                "rgb_img": rgb_img,
                "gt_R": gt_R,
                "gt_S": gt_S,
                "mask": mask,
                "index": index,
                "img_name": filename,
                "dataset": self.dataset_name}


def check_dataset_split(dataset_mit, batch_size, num_workers, disp_iters, visualize_dir=None) -> None:
    # from utils.image_util import plot_images

    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

    data_loader_mit = torch.utils.data.DataLoader(dataset_mit,
                                                  shuffle=False, drop_last=False,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
    end = time.time()
    for index, data_mit in enumerate(data_loader_mit):
        sample = data_mit["srgb_img"]
        if (index + 1) % disp_iters == 0:
            print(f"index: {index}, image shape: {sample.shape}, "
                  f"time/batch: {(time.time()-end)/disp_iters}")
            end = time.time()
        if visualize_dir is not None and ((index+1) % disp_iters == 0):
            for b in range(sample.size(0)):
                # vis_imgs = {k: v[b] for k, v in data_mit.items()
                #             if k not in ["index", "dataset", "img_name"]}
                # vis_imgs = {k: v.repeat(3, 1, 1) if v.size(0) == 1 else v for k, v in vis_imgs.items()}
                # _imgs = [v for k, v in vis_imgs.items()]
                # _titles = [k for k, v in vis_imgs.items()]
                # plt = plot_images(_imgs, _titles, columns=4, show=False)
                # plt.savefig(os.path.join(visualize_dir, f"{data_mit['img_name'][b]}.jpeg"))
                # plt.close()
                vis_imgs = [v[b] for k, v in data_mit.items()
                            if k not in ["index", "dataset", "img_name"]]
                torchvision.utils.save_image(vis_imgs,
                                             os.path.join(visualize_dir, f"{data_mit['img_name'][b]}.jpeg"))


def is_MIT_Intrinsic_complete(dataset_dir: str, batch_size=8, num_workers=1, visualize_dir=None) -> bool:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        train_sample_dir = os.path.join(visualize_dir, "train_samples")
        test_sample_dir = os.path.join(visualize_dir, "test_samples")
    else:
        train_sample_dir = test_sample_dir = None

    dataset_mit_train, dataset_mit_test = get_train_test_sets(dataset_dir, (224, 224))
    print(f"MIT _ train:\n"
          f"\tsize: {len(dataset_mit_train)}, batch_size: {batch_size}")
    check_dataset_split(dataset_mit_train, batch_size, num_workers, 1, train_sample_dir)

    print(f"MIT _ test:\n"
          f"\tsize: {len(dataset_mit_test)}, batch_size: {batch_size}")
    check_dataset_split(dataset_mit_test, batch_size, num_workers, 1, test_sample_dir)
    return True


def get_train_test_sets(dataset_dir: str, train_img_size=None):
    return MITIntrinsicDataset(dataset_dir, "train", img_size=train_img_size), \
           MITIntrinsicDataset(dataset_dir, "test")


