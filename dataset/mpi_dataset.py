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
import random

import torch, torchvision
import torch.utils.data as data
import numpy as np
# from skimage import io
import skimage
from torchvision import io
from torchvision.transforms import RandomResizedCrop
# from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F
from kornia.geometry import resize

import constants as C


class MPIDataset(data.Dataset):
    """
    Load data from MPI dataset
    """
    dataset_name = C.MPI_Sintel
    input_image_dir = "MPI-main-clean"
    gt_R_dir = "MPI-main-albedo"
    gt_S_dir = "MPI-main-shading"
    mask_dir = "MPI-main-mask"
    split_file_dir = "split_files"
    split_files = {
        "image": {
            "front": {
                "train": "MPI_main_imageSplit-fullsize-ChenSplit-train.txt",
                "test": "MPI_main_imageSplit-fullsize-ChenSplit-test.txt",
            },
            "reverse": {
                "test": "MPI_main_imageSplit-fullsize-ChenSplit-train.txt",
                "train": "MPI_main_imageSplit-fullsize-ChenSplit-test.txt",
            }
        },
        "scene": {
            "front": {
                "train": "MPI_main_sceneSplit-fullsize-NoDefect-train.txt",
                "test": "MPI_main_sceneSplit-fullsize-NoDefect-test.txt",
            },
            "reverse": {
                "test": "MPI_main_sceneSplit-fullsize-NoDefect-train.txt",
                "train": "MPI_main_sceneSplit-fullsize-NoDefect-test.txt",
            }
        }
    }

    def __init__(self, root: str,
                 split: str,
                 mode: str,
                 img_size=None,
                 ) -> None:
        loc = split.find("_")
        split_type, fold_order = split[:loc], split[loc+1:]
        assert split_type in ["image", "scene"]
        assert fold_order in ["front", "reverse"]
        assert mode in ["train", "test", "val"]
        if mode == "val":
            mode = "test"
        self.mode = mode
        self.is_train = (self.mode in ["train"])
        assert (img_size is None) or self.is_train, f"Don't resize {mode} images."
        self.img_size = img_size

        # check dataset path
        self.root = root
        self.input_image_dir = os.path.join(self.root, self.input_image_dir)
        self.gt_R_dir = os.path.join(self.root, self.gt_R_dir)
        self.gt_S_dir = os.path.join(self.root, self.gt_S_dir)
        self.mask_dir = os.path.join(self.root, self.mask_dir)
        self.split_file_dir = os.path.join(self.root, self.split_file_dir)
        self.data_list = self._get_data_list(os.path.join(self.split_file_dir,
                                                          self.split_files[split_type][fold_order][self.mode]))

    def __len__(self):
        return len(self.data_list)

    def _get_data_list(self, path):
        flag = self._check_exists([
            self.root,
            self.input_image_dir,
            self.gt_R_dir,
            self.gt_S_dir,
            self.mask_dir,
            self.split_file_dir,
            path
        ])
        if not flag:
            raise RuntimeError(f"MPI dataset is not found or not complete "
                               f"in the path: {self.root}")
        # load list
        with open(path) as f:
            data_list = f.readlines()
        return data_list

    def _check_exists(self, paths) -> bool:
        flag = True
        for p in paths:
            flag = flag and os.path.exists(p)
        return flag

    def load_images(self, path: str, augment_data: bool):

        # load images
        _, filename = os.path.split(path.strip())
        srgb_img_path = os.path.join(self.input_image_dir, filename)
        gt_R_path = os.path.join(self.gt_R_dir, filename)
        gt_S_path = os.path.join(self.gt_S_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        srgb_img = io.read_image(srgb_img_path).to(torch.float32) / 255.0
        gt_R = io.read_image(gt_R_path).to(torch.float32) / 255.0
        gt_S = io.read_image(gt_S_path).to(torch.float32) / 255.0
        mask = np.float32(skimage.io.imread(mask_path)) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).repeat(gt_R.size(0), 1, 1)

        # set mask
        # mask[np.mean(gt_R, 2) < 1e-5] = 0
        # mask = (mask > 0.5).astype(np.float32)
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.repeat(mask, 3, axis=2)
        # mask = torch.ones_like(gt_R)

        # data augmentation
        if augment_data:
            data_tuple = (srgb_img, gt_R, gt_S, mask)
            if self.img_size is not None:
                # assert self.img_size[0] <= srgb_img.size(1)
                # assert self.img_size[1] <= srgb_img.size(2)
                # h_crop = random.randint(self.img_size[0], srgb_img.size(1))
                # w_crop = random.randint(self.img_size[1], srgb_img.size(2))
                # i, j, th, tw = RandomCrop.get_params(srgb_img, (h_crop, w_crop))
                # data_tuple = (F.crop(d, i, j, th, tw) for d in data_tuple)
                # data_tuple = (resize(d, size=self.img_size, interpolation="area", antialias=True)
                #               for d in data_tuple)
                i, j, h, w = RandomResizedCrop.get_params(srgb_img, scale=[0.1, 1.0], ratio=[4. / 10., 10. / 4.])
                data_tuple = (F.crop(d, i, j, h, w) for d in data_tuple)
                data_tuple = (resize(d, size=self.img_size, interpolation="area", antialias=True)
                              for d in data_tuple)
            if torch.rand(1) < 0.5:
                data_tuple = (F.hflip(d) for d in data_tuple)
            if torch.rand(1) < 0.5:
                data_tuple = (F.vflip(d) for d in data_tuple)
            srgb_img, gt_R, gt_S, mask = data_tuple
        mask = (mask > 0.99).to(torch.float32)
        gt_R = gt_R * mask
        gt_S[mask == 0] = 0.5
        rgb_img = srgb_img ** 2.2
        return srgb_img, rgb_img, gt_R, gt_S, mask, filename[:-4]

    def numpy_images_2_tensor(self, *imgs):
        out = (torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).contiguous().to(torch.float32)
               for img in imgs)
        return out

    def __getitem__(self, index):
        srgb_img, rgb_img, gt_R, gt_S, mask, filename = self.load_images(self.data_list[index], self.is_train)
        # srgb_img, rgb_img, gt_R, gt_S, mask = self.numpy_images_2_tensor(srgb_img, rgb_img, gt_R, gt_S, mask)
        return {"srgb_img": srgb_img,
                "rgb_img": rgb_img,
                "gt_R": gt_R,
                "gt_S": gt_S,
                "mask": mask,
                "index": index,
                "img_name": filename,
                "dataset": self.dataset_name}


def check_dataset_split(dataset_mpi, batch_size, num_workers, disp_iters, visualize_dir=None) -> None:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

    data_loader_mpi = torch.utils.data.DataLoader(dataset_mpi,
                                                  shuffle=False, drop_last=False,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
    end = time.time()
    for index, data_mpi in enumerate(data_loader_mpi):
        sample = data_mpi["srgb_img"]
        if (index + 1) % disp_iters == 0:
            print(f"index: {index}, image shape: {sample.shape}, "
                  f"time/batch: {(time.time()-end)/disp_iters}")
            end = time.time()
        if visualize_dir is not None and ((index+1) % disp_iters == 0):
            for b in range(sample.size(0)):
                vis_imgs = [v[b] for k, v in data_mpi.items()
                            if k not in ["index", "dataset", "img_name"]]
                torchvision.utils.save_image(vis_imgs,
                                             # os.path.join(visualize_dir, f"{index}-{b}-{data_mpi['img_name'][b]}.jpeg"))
                                             os.path.join(visualize_dir, f"{data_mpi['img_name'][b]}.jpeg"))


def is_MPI_complete(dataset_dir: str, batch_size=8, num_workers=1, visualize_dir=None) -> bool:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        train_sample_dir = os.path.join(visualize_dir, "train_samples")
        test_sample_dir = os.path.join(visualize_dir, "test_samples")
    else:
        train_sample_dir = test_sample_dir = None

    dataset_mpi_train, dataset_mpi_test = get_train_test_sets(dataset_dir, "scene_reverse")
    print(f"MPI _ train:\n"
          f"\tsize: {len(dataset_mpi_train)}, batch_size: {batch_size}")
    check_dataset_split(dataset_mpi_train, batch_size, num_workers, 1, train_sample_dir)

    print(f"MPI _ test:\n"
          f"\tsize: {len(dataset_mpi_test)}, batch_size: {batch_size}")
    check_dataset_split(dataset_mpi_test, batch_size, num_workers, 1, test_sample_dir)
    return True


def get_train_test_sets(dataset_dir: str, split: str):
    return MPIDataset(dataset_dir, split, "train", img_size=(392, 392)), \
           MPIDataset(dataset_dir, split, "test", img_size=None)


