import os
import os.path
import time
import random

import torch, torchvision
import torch.utils.data as data
# from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F
import numpy as np
from skimage import io
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
# from PIL import Image
from kornia.geometry import resize

from utils import image_util
import constants as C


class CGIntrinsicsDataset(data.Dataset):
    """
    Load data from CGIntrinsics dataset
    """
    images_dir = "images"
    train_list = "train_test_split/CGI_train_split_v0.csv"
    val_list = None  # undefined
    test_list = "train_test_split/CGI_test_split_v0.csv"
    rgbe_image_stats_file = "rgbe_image_stats.txt"
    dataset_name = C.CGIntrinsics

    def __init__(self, root: str,
                 mode: str,
                 img_size: tuple or None,
                 require_linear_input=False) -> None:
        assert mode in ["train", "full"] or img_size is None, f"Not resize image for test."

        self.mode = mode
        self.is_train = (self.mode in ["train", "full"])
        self.img_size = img_size
        self.require_linear_input = require_linear_input

        # check dataset path
        self.root = root
        self.images_dir = os.path.join(self.root, self.images_dir)
        if self.mode == "train":
            self.data_list = self._get_data_list(os.path.join(self.root, self.train_list))
        # elif self.mode == "val":
        #     assert self.val_list is not None, "Undefined val_list."
        #     self.list_file = os.path.join(self.root, self.val_list)
        elif self.mode == 'test':
            self.data_list = self._get_data_list(os.path.join(self.root, self.test_list))
        elif self.mode == "full":
            train_split = self._get_data_list(os.path.join(self.root, self.train_list))
            test_split = self._get_data_list(os.path.join(self.root, self.test_list))
            self.data_list = pd.concat([train_split, test_split], ignore_index=True)
        else:
            raise Exception(f"CGIntrinsicsDataset only supports mode: {'train', 'test', 'full'}")

        # rgbe_image_stats
        self.rgbe_image_stats_file = os.path.join(self.root, self.rgbe_image_stats_file)
        self.stat_dict = {}
        f = open(self.rgbe_image_stats_file, "r")
        line = f.readline()
        while line:
            line = line.split()
            self.stat_dict[line[0]] = float(line[2])
            line = f.readline()

    def __len__(self):
        return len(self.data_list)

    def _get_data_list(self, path):
        flag = self._check_exists([
            self.root,
            self.images_dir,
            path
        ])
        if not flag:
            raise RuntimeError(f"CGIntrinsics dataset is not found or not complete "
                               f"in the path: {self.root}")
        # load csv list
        data_list = pd.read_csv(path, header=None)
        return data_list

    def _check_exists(self, paths) -> bool:
        flag = True
        for p in paths:
            flag = flag and os.path.exists(p)
        return flag

    def load_images(self, path: str, augment_data: bool):

        # image paths
        img_dir, filename = os.path.split(path)
        len_postfix = len(".png")
        srgb_img_path = os.path.join(self.images_dir, path)
        gt_R_path = os.path.join(self.images_dir, img_dir, filename[:-len_postfix] + "_albedo.png")
        mask_path = os.path.join(self.images_dir, img_dir, filename[:-len_postfix] + "_mask.png")

        # load images
        srgb_img = np.float32(io.imread(srgb_img_path)) / 255.0
        gt_R = np.float32(io.imread(gt_R_path)) / 255.0
        mask = np.float32(io.imread(mask_path)) / 255.0

        # set mask
        mask[np.mean(gt_R, 2) < 1e-5] = 0
        # mask[np.mean(srgb_img, 2) < 1e-6] = 0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)

        # to Tensor
        srgb_img, gt_R, mask = self.numpy_images_2_tensor(srgb_img, gt_R, mask)
        gt_R = image_util.adjust_image_for_display(gt_R, True, False, 0.9999, 0.95, True)
        rgb_img = srgb_img ** 2.2
        # gt_S
        gt_S = rgb_img / gt_R.clamp(min=1e-5) * mask
        search_name = path[:-4] + ".rgbe"
        irridiance = self.stat_dict[search_name]
        if irridiance < 0.25:
            gt_S = gt_S.permute(1, 2, 0).numpy()
            gt_S = denoise_tv_chambolle(gt_S, weight=0.1, multichannel=True)
            gt_S = next(self.numpy_images_2_tensor(gt_S))
        # gt_S = gt_S.mean(axis=2, keepdims=True).repeat(3, axis=2)

        data_tuple = (srgb_img, rgb_img, gt_R, gt_S, mask)
        # data augmentation
        if augment_data:
            # i, j, h, w = RandomResizedCrop.get_params(srgb_img, scale=[0.5, 1.0], ratio=[1., 1.])
            # data_tuple = (F.resized_crop(d, i, j, h, w, self.img_size, Image.BILINEAR) for d in data_tuple)
            if random.random() > 0.5:
                data_tuple = (F.hflip(d) for d in data_tuple)
            # if torch.rand(1) < 0.5:
            #     data_tuple = (F.vflip(d) for d in data_tuple)
        # resize images
        if self.img_size is not None:
            data_tuple = (resize(d, size=self.img_size, interpolation="area", antialias=True)
                          for d in data_tuple)
        srgb_img, rgb_img, gt_R, gt_S, mask = data_tuple

        mask = (mask > 0.99).to(torch.float32)
        gt_R = gt_R * mask
        if self.require_linear_input:
            gt_S[mask == 0] = 0.0
            mask_saturated = (srgb_img.max(dim=0, keepdim=True)[0] > 0.99).to(torch.float32)
            mask *= (1.0 - mask_saturated)
            # gt_S *= image_util.get_scale_alpha(gt_S, mask, 0.9, 0.8)
            # gt_S = gt_S.clamp(min=0.0, max=1.0)
            srgb_img = rgb_img
        else:
            gt_S[mask == 0] = 0.5
        gt_S = gt_S.mean(dim=0, keepdim=True).repeat(3, 1, 1)  # gray shading
        return srgb_img, rgb_img, gt_R, gt_S, mask, f"{img_dir}_{filename[:-len_postfix]}"

    def numpy_images_2_tensor(self, *imgs):
        out = (torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).contiguous().to(torch.float32)
               for img in imgs)
        return out

    def __getitem__(self, index):
        srgb_img, rgb_img, gt_R, gt_S, mask, filename = self.load_images(self.data_list[0][index], self.is_train)
        return {"srgb_img": srgb_img,
                "rgb_img": rgb_img,
                "gt_R": gt_R,
                "gt_S": gt_S,
                # "gt_S_srgb": gt_S ** (1/2.2),
                "mask": mask,
                "index": index,
                "img_name": filename,
                "dataset": self.dataset_name}


def check_dataset_split(dataset_cgi, batch_size, num_workers, disp_iters, visualize_dir=None) -> None:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

    data_loader_cgi = torch.utils.data.DataLoader(dataset_cgi,
                                                  shuffle=False, drop_last=False,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
    end = time.time()
    for index, data_cgi in enumerate(data_loader_cgi):
        sample = data_cgi["srgb_img"]
        if (index + 1) % disp_iters == 0:
            print(f"index: {index}, image shape: {sample.shape}, "
                  f"time/batch: {(time.time()-end)/disp_iters}")
            end = time.time()
        if visualize_dir is not None and ((index+1) % (1000//batch_size) == 0):
            vis_imgs = [v[0] for k, v in data_cgi.items()
                        if k not in ["index", "dataset", "img_name"]]
            torchvision.utils.save_image(vis_imgs,
                                         os.path.join(visualize_dir, f"{index}-0.jpeg"))


def is_CGIntrinsics_complete(dataset_dir: str, batch_size=8, num_workers=1, visualize_dir=None) -> bool:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        train_sample_dir = os.path.join(visualize_dir, "train_samples")
        val_sample_dir = os.path.join(visualize_dir, "val_samples")
        test_sample_dir = os.path.join(visualize_dir, "test_samples")
    else:
        train_sample_dir = val_sample_dir = test_sample_dir = None

    dataset_cgi_train = CGIntrinsicsDataset(dataset_dir, "train", (224, 224))
    print(f"CGIntrinsics _ train:\n"
          f"\tsize: {len(dataset_cgi_train)}, batch_size: {batch_size}")
    check_dataset_split(dataset_cgi_train, batch_size, num_workers, 10, train_sample_dir)

    # dataset_cgi_val = CGIntrinsicsDataset(dataset_dir, "val")
    # print(f"CGIntrinsics _ val:\n"
    #       f"\tsize: {len(dataset_cgi_val)}, batch_size: {batch_size}")
    # check_dataset_split(dataset_cgi_val, batch_size, num_workers, 10, val_sample_dir)

    dataset_cgi_test = CGIntrinsicsDataset(dataset_dir, "test", None)
    print(f"CGIntrinsics _ test:\n"
          f"\tsize: {len(dataset_cgi_test)}, batch_size: {batch_size}")
    check_dataset_split(dataset_cgi_test, batch_size, num_workers, 10, test_sample_dir)
    return True


# def get_train_val_test_sets(dataset_dir: str):
#     return CGIntrinsicsDataset(dataset_dir, "train"), CGIntrinsicsDataset(dataset_dir, "val"), \
#            CGIntrinsicsDataset(dataset_dir, "test")


