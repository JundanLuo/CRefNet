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


import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os


def filter_img_files(filepath):
    return filepath.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))

def filter_img_files_exclude_mask(filepath):
    is_img = filter_img_files(filepath)
    return is_img and \
           not filepath.lower().endswith(("mask.png", "mask.jpg", "mask.jpeg", "mask.tif"))


def filter_mit_test_files(filepath):
    return filepath.lower().endswith("-input.png")


class ImageFolderLoader(Dataset):
    def __init__(self, data_dir, mode=None):
        print(f"Dataloader: {data_dir} folder, {mode} mode.")
        self.data_dir = data_dir
        self.dataset_name = self.data_dir
        self.mode = mode
        img_filter = {
            "none": filter_img_files,
            "with_mask": filter_img_files_exclude_mask,
            "MIT_test": filter_mit_test_files,
        }[mode]
        self.data_list = self.list_files(self.data_dir, img_filter)

    def list_files(self, directory, img_filter):
        """List all files in a directory, filtered by img_filter."""
        file_list = os.listdir(directory)
        file_list = [os.path.join(directory, f) for f in file_list if img_filter(f)]
        return file_list

    def read_image(self, path: str, type: str):
        """ Read image from path
        """
        MAX_8bit = 255.0
        MAX_16bit = 65535.0
        # Read image
        assert os.path.exists(path), f"Image {path} does not exist."
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Read image {path} failed."
        # Convert to float32 [0, 1]
        if img.dtype == np.uint16:  # Note that MIT images are 16 bits
            img = img.astype(np.float32) / MAX_16bit
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / MAX_8bit
        else:
            raise NotImplementedError(f"Image type {img.dtype} is not implemented.")
        # Check image shape and convert to RGB
        assert img.ndim < 4, f"Image should be 2D or 3D, but got {img.ndim}."
        if img.ndim == 3:
            img = img[:, :, ::-1]  # BGR -> RGB
            assert img.shape[-1] == 3 or img.shape[-1] == 1, \
                f"Image should be RGB or gray-scale, but got {img.shape}."
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # HW -> HW1
        # Convert to specified type
        if type == "numpy":
            pass
        elif type == "tensor":
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = torch.from_numpy(img.copy()).contiguous()
        else:
            raise NotImplementedError(f"Type {type} is not implemented.")
        return img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Path
        img_path = self.data_list[idx]
        filename_with_ext = os.path.basename(img_path)  # Get filename with extension
        filename_without_ext = os.path.splitext(filename_with_ext)[0]  # Get filename without extension
        # Read image
        input_img = self.read_image(img_path, "tensor")  # [C, H, W]
        gt_R = gt_S = None
        if self.mode in ["none", "with_mask"]:
            filename = filename_without_ext
            if self.mode in ["none"]:
                mask = torch.ones_like(input_img)  # [C, H, W]
            elif self.mode == "with_mask":
                mask_path = os.path.join(self.data_dir, filename + "_mask.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(self.data_dir, filename + "_mask.jpg")
                    if not os.path.exists(mask_path):
                        assert False, f"Mask {mask_path} does not exist."
                mask = self.read_image(mask_path, "tensor")
                mask = (mask > 0.5).to(torch.float32)  # [C, H, W]
        elif self.mode == "MIT_test":
            filename = filename_without_ext[:-len("-input")]
            mask_path = os.path.join(self.data_dir, filename + "-label-mask.png")
            mask = self.read_image(mask_path, "tensor")  # [C, H, W]
            mask = (mask > 0.5).to(torch.float32)  # [C, H, W]
            # gt_R_path = os.path.join(self.data_dir, filename + "-label-albedo.png")
            # gt_S_path = os.path.join(self.data_dir, filename + "-label-shading.png")
            # gt_R = self.read_image(gt_R_path, "tensor")
            # gt_S = self.read_image(gt_S_path, "tensor").repeat(3, 1, 1)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented.")
        mask = mask.expand_as(input_img)  # [C, H, W]
        data_dict = {
            "img_path": img_path,
            "img_name": filename,
            "srgb_img": input_img,
            "mask": mask,
            "index": idx,
            "dataset": self.dataset_name,
        }
        if gt_R is not None:
            data_dict["gt_R"] = gt_R
        if gt_S is not None:
            data_dict["gt_S"] = gt_S
        return data_dict
