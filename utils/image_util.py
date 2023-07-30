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


import io
import os
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2


def rgb_to_srgb(rgb, gamma=1.0/2.2):
    return rgb.clip(min=0.0) ** gamma


def srgb_to_rgb(srgb, gamma=1.0/2.2):
    return srgb.clip(min=0.0) ** (1.0 / gamma)


def valid_rgI(rgI, v_min=0.0, v_max=1.0):
    assert rgI.ndim == 4 and rgI.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgI)
    for i in range(rgI.size(1)):
        c = rgI[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    s = rgI[:, 0:1, :, :] + rgI[:, 1:2, :, :]
    ret = ret.mul(s >= v_min).mul(s <= v_max)
    return ret


def valid_rgb(rgb, v_min=0.0, v_max=1.0):
    assert rgb.ndim == 4 and rgb.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgb)
    for i in range(rgb.size(1)):
        c = rgb[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    if rgb.ndim == 3:
        sum = torch.sum(rgb, dim=0, keepdim=True).clamp(min=1e-6)
    elif rgb.ndim == 4:
        sum = torch.sum(rgb, dim=1, keepdim=True).clamp(min=1e-6)
    else:
        raise Exception("Only supports image: [C, H, W] or [B, C, H, W]")
    chromat = rgb / sum
    return chromat


def save_srgb_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.to(torch.float32).cpu().numpy(), (1, 2, 0)) * 255.0
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np, mode='RGB')
    # Save Image
    if not os.path.exists(path):
        os.makedirs(path)
    image_pil.save(os.path.join(path,filename))


def tone_mapping(image, rescale=True, trans2srgb=False):
    # MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837
    src_PERCENTILE = 0.9
    dst_VALUE = 0.8

    vis = image.detach()
    if vis.size(0) == 1:
        vis = vis.repeat(3, 1, 1)

    if rescale:
        brightness = 0.3 * vis[0, :, :] + 0.59 * vis[1, :, :] + 0.11 * vis[2, :, :]
        src_value = brightness.quantile(src_PERCENTILE)
        if src_value < 1.0e-4:
            scalar = 0.0
        else:
            scalar = math.exp(math.log(dst_VALUE) * 2.2 - math.log(src_value))
        vis = scalar * vis
        # s = np.percentile(vis.cpu(), 99.9)
        # # if mask is None:
        # #     s = np.percentile(vis.numpy(), 99.9)
        # # else:
        # #     s = np.percentile(vis[mask > 0.5].numpy(), 99.9)
        # if s > MAX_SRGB:
        #     vis = vis / s * MAX_SRGB

    vis = torch.clamp(vis, min=0)
    if trans2srgb:
        # vis[vis > MAX_SRGB] = MAX_SRGB
        vis = rgb_to_srgb(vis)

    vis = vis.clamp(min=0.0, max=1.0)
    return vis


def adjust_image_for_display(image: torch.tensor, rescale: bool, trans2srgb: bool, src_percentile=0.9999, dst_value=0.85,
                             clip: bool = True):
    assert image.ndim == 3 or image.ndim == 4, "Only supports image: [C, H, W] or [B, C, H, W]"
    vis = image.detach()
    if vis.ndim == 3:
        vis = vis.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    if vis.size(1) == 1:
        vis = vis.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]

    if rescale:
        src_value = vis.mean(dim=1, keepdim=True)
        src_value = src_value.view(src_value.size(0), -1).quantile(src_percentile, dim=1)
        src_value[src_value < 1e-5] = 1.0
        vis = vis / src_value.view(src_value.size(0), 1, 1, 1) * dst_value
    if trans2srgb:
        vis = rgb_to_srgb(vis)

    if image.ndim == 3:
        vis = vis.squeeze(0)
    return vis.clamp(min=0.0, max=1.0) if clip else vis


def get_scale_alpha(image: torch.tensor, src_percentile: float, dst_value: float):
    assert image.ndim == 3 or image.ndim == 4, "Only supports image: [C, H, W] or [B, C, H, W]"
    vis = image.detach()
    if vis.ndim == 3:
        vis = vis.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    src_value = vis.mean(dim=1, keepdim=True)
    src_value = src_value.view(src_value.size(0), -1).quantile(src_percentile, dim=1)
    src_value[src_value < 1e-5] = 1.0
    alpha = 1.0 / src_value.view(src_value.size(0), 1, 1, 1).clamp(min=1e-5) * dst_value
    if image.ndim == 3:
        alpha = alpha.squeeze(0)
    return alpha


def convert_plot_to_tensor(plot):
    """convert plt to tensor."""
    buf = io.BytesIO()
    plot.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    t = ToTensor()(image)
    return t


def generate_histogram_image(array, title, xlabel, ylabel, bins=100, range=None):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    plt.hist(array, histtype='bar', alpha=0.3, bins=bins, range=range)
    hist_img = convert_plot_to_tensor(plt)
    return hist_img


def numpy_to_tensor(img):
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        assert False
    return torch.from_numpy(img).contiguous().to(torch.float32)


def tensor_to_numpy(img):
    img = img.cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    else:
        assert False
    return img


def split_tenors_from_dict(dict, size):
    a, b = {}, {}
    for key, item in dict.items():
        if isinstance(item, torch.Tensor):
            a[key], b[key] = item.split(size, dim=0)
        else:
            a[key] = b[key] = item
    return a, b


def plot_images(_images, titles=None, figsize_base=4, columns=3, show=True):
    num_images = len(_images)
    rows = (num_images + columns - 1) // columns
    figsize = (figsize_base * columns, int(figsize_base * rows * 0.8))
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axs = np.array(axs).ravel()  # Make sure axs is always a 1D array
    for i, img in enumerate(_images):
        if torch.is_tensor(img):
            img = img.numpy().transpose(1, 2, 0)
        if img.dtype == np.float32:
            img = img.clip(min=0.0, max=1.0)
        elif img.dtype == np.uint8:
            img = img.clip(min=0, max=255)
        axs[i].imshow(img, cmap="gray")
        if titles is not None:
            axs[i].set_title(titles[i])
    for i in range(len(_images), rows * columns):
        axs[i].axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return plt


def read_image(path: str, type: str):
    """ Read image from path """
    MAX_8bit = 255.0
    MAX_16bit = 65535.0
    # Read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # Convert to float32 [0, 1]
    if img.dtype == np.uint16:
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
