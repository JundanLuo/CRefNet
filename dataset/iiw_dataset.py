import os
import re
import collections

import torch.utils.data
import pickle
import numpy as np
import torch
from torch._six import string_classes
import random
from skimage import io
from skimage.transform import resize
import h5py

from utils.image_util import srgb_to_rgb
import constants as C


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg


def make_dataset(list_dir):
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ), encoding='iso-8859-1')
    return images_list


class IIWDataset(torch.utils.data.Dataset):
    test_list = "test_list/img_batch.p"
    train_list = "train_list/img_batch.p"
    dataset_name = C.IIW

    def __init__(self, root, ori_idx, mode, img_size=None, pseudo_gt_dir=None):
        assert mode in ["test", "train", "val"]
        self.is_test = mode in ["test", "val"]
        assert not (img_size is not None and self.is_test), f"Don't resize {mode} images."
        self.root = root
        self.current_o_idx = ori_idx
        # self.set_o_idx(ori_idx)
        self.img_size = img_size
        self.mode = mode
        self.pseudo_gt_dir = pseudo_gt_dir
        if self.pseudo_gt_dir is not None:
            self.pseudo_gt_dir = os.path.join(self.pseudo_gt_dir,
                                              "pred_" + {"train": "train",
                                                         "test": "test",
                                                         "val": "train",
                                                         }[self.mode])

        # load image list
        list_file = {
            "train": os.path.join(self.root, self.train_list),
            "test": os.path.join(self.root, self.test_list),
            "val": os.path.join(self.root, self.train_list)
        }[self.mode]
        if not self._check_exists([self.root,
                                   self.pseudo_gt_dir,
                                   os.path.join(self.root, "data"),
                                   os.path.join(self.root, "long_range_data_4"),
                                   list_file]):
            raise RuntimeError(f"IIW dataset is not found or not complete in the path: {self.root}")
        if self.current_o_idx in [0, 1, 2]:  # one orientation
            self.img_list = self._get_img_list(list_file, self.current_o_idx, self.mode)
        elif self.current_o_idx == -1:  # all orientation
            self.img_list = []
            for o in [0, 1, 2]:
                self.img_list = self.img_list + self._get_img_list(list_file, o, self.mode)
        else:
            assert False, "ori_idx must be in [-1, 0, 1, 2]."
        self.num_scale = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.half_window = 1
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)

    def _get_resize_shape(self, o_idx):
        if o_idx == 0:
            # self.height = 256
            # self.width = 384
            return 256, 384
        elif o_idx == 1:
            # self.height = 384
            # self.width = 256
            return 384, 256
        elif o_idx == 2:
            # self.height = 384
            # self.width = 384
            return 384, 384
        # elif o_idx == 3:
        #     self.height = 384
        #     self.width = 512
        # else:
        #     self.height = 512
        #     self.width = 384
        else:
            assert False, f"Error o_idx: {o_idx}"

    # def DA(self, img, mode, random_filp):
    #
    #     # if random_filp > 0.5:
    #     # img = np.fliplr(img)
    #
    #     # img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
    #     if self.target_img_size is not None:
    #         img = resize(img, self.target_img_size, order=mode, mode='constant', anti_aliasing=True)
    #
    #     return img

    def _get_img_list(self, list_file, ori_idx, mode):
        img_list = pickle.load(open(list_file, "rb"), encoding='iso-8859-1')[ori_idx]
        assert len(img_list) > 0, f"Found 0 images in {list_file}"
        if mode == "train":
            img_list = [img_list[i] for i in range(len(img_list)) if i % 5 != 0]
        elif mode == "val":
            img_list = img_list[::5]
        return img_list

    def _check_exists(self, paths) -> bool:
        flag = True
        for p in paths:
            if p is None:
                continue
            flag = flag and os.path.exists(p)
        return flag

    def iiw_loader(self, img_path, gt_s_path, data_augment):
        img = np.float32(io.imread(img_path)) / 255.0
        if gt_s_path is not None:
            gt_S = np.load(gt_s_path).astype(np.float32)
            assert gt_S.ndim == 3 and gt_S.shape[2] == 3, f"gt_S shape {img.shape}, {gt_S.shape}"
        else:
            gt_S = np.zeros_like(img)
        original_shape = img.shape
        if self.img_size is not None:
            img = resize(img, self.img_size, order=1, mode='constant', anti_aliasing=True)
        gt_S = resize(gt_S, img.shape[:-1], order=1, mode='constant', anti_aliasing=True)  # NIID-Net pred is not of the same size as input
        is_flipped = False
        if data_augment:
            is_flipped = random.random() > 0.5
            if is_flipped:
                img = np.fliplr(img)
                gt_S = np.fliplr(gt_S)
        # mask = ((gt_S > 1e-4) * (gt_S < 20.0)).astype(np.float32)
        mask = np.ones_like(img)
        return img, gt_S, mask, is_flipped, original_shape

    def construst_R_weights(self, N_feature):

        center_feature = np.repeat( np.expand_dims(N_feature[4, :, :,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - N_feature

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:2]**2  , 3) / (self.sigma_chro**2)) \
              * np.exp(- (feature_diff[:,:,:,2]**2) /(self.sigma_I**2) )

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                                      self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :]
                ct_idx += 1

        return sub_C

    def long_range_loader(self, h5_path):
        hdf5_file_read_img = h5py.File(h5_path,'r')
        num_eq = hdf5_file_read_img.get('/info/num_eq')
        num_eq = np.float32(np.array(num_eq))
        num_eq = int(num_eq[0][0])

        if num_eq > 0:
            equal_mat = hdf5_file_read_img.get('/info/equal')
            equal_mat = np.float32(np.array(equal_mat))
            equal_mat = np.transpose(equal_mat, (1, 0))
            # equal_mat = torch.from_numpy(equal_mat).contiguous().float()
        else:
            # equal_mat = torch.Tensor(1, 1)
            equal_mat = np.zeros((1, 1))

        num_ineq = hdf5_file_read_img.get('/info/num_ineq')
        num_ineq = np.float32(np.array(num_ineq))
        num_ineq = int(num_ineq[0][0])

        if num_ineq > 0:
            ineq_mat = hdf5_file_read_img.get('/info/inequal')
            ineq_mat = np.float32(np.array(ineq_mat))
            ineq_mat = np.transpose(ineq_mat, (1, 0))
            # ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
        else:
            # ineq_mat = torch.Tensor(1, 1)
            ineq_mat = np.zeros((1, 1))

        hdf5_file_read_img.close()
        return equal_mat, ineq_mat

    def __getitem__(self, index):
        img_idx = self.img_list[index].split('/')[-1][:-7]
        # img_path = os.path.join(self.root, self.img_list[self.current_o_idx][index]).split('/')[-1]
        judgement_path = os.path.join(self.root, "data", img_idx + '.json')
        mat_path = os.path.join(self.root, "long_range_data_4", img_idx + '.h5')
        img_path = os.path.join(self.root, "data", img_idx+".png")
        if self.pseudo_gt_dir is not None:
            gt_s_path = os.path.join(self.pseudo_gt_dir, "raw", f"{img_idx}-s.npy")
        else:
            gt_s_path = None


        # img, random_filp = self.iiw_loader(img_path)
        srgb_img, gt_S, mask, is_flipped, original_shape = self.iiw_loader(img_path, gt_s_path, not self.is_test)

        targets_1 = {}
        targets_1['mat_path'] = mat_path
        targets_1['path'] = img_path
        targets_1["judgements_path"] = judgement_path
        targets_1["random_flip"] = is_flipped
        targets_1["original_shape"] = original_shape
        targets_1["original_h"] = original_shape[0]
        targets_1["original_w"] = original_shape[1]

        # if random_filp > 0.5:
        # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch_flip/" + img_path.split('/')[-1] + "/R0.h5"
        # else:
        # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch/" + img_path.split('/')[-1] + "/R0.h5"

        rgb_img = srgb_to_rgb(srgb_img)
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()
        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()

        # for i in range(0, self.num_scale):
        #     feature_3d = rgb_to_irg(rgb_img)
        #     sub_matrix = self.construst_sub_matrix(feature_3d)
        #     r_w = self.construst_R_weights(sub_matrix)
        #     targets_1['r_w_s'+ str(i)] = torch.from_numpy(r_w).float()
        #     rgb_img = rgb_img[::2,::2,:]

        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(srgb_img, (2,0,1)))).contiguous().float()
        gt_S = torch.from_numpy(np.ascontiguousarray(np.transpose(gt_S, (2,0,1)))).contiguous().float()
        mask = torch.from_numpy(np.ascontiguousarray(np.transpose(mask, (2,0,1)))).contiguous().float()

        # sparse_shading_name = str(self.height) + "x" + str(self.width)
        #
        # if self.current_o_idx == 0:
        #     sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        # elif self.current_o_idx == 1:
        #     sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name +  "/R0.h5"
        # elif self.current_o_idx == 2:
        #     sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        # elif self.current_o_idx == 3:
        #     sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        # elif self.current_o_idx == 4:
        #     sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"

        if not self.is_test:
            eq_mat, ineq_mat = self.long_range_loader(mat_path)
            targets_1["eq_mat"] = eq_mat
            targets_1["ineq_mat"] = ineq_mat

        out = {"srgb_img": final_img,
                "rgb_img": targets_1["rgb_img"],
                "targets": targets_1,
                "mask": mask,
                # "sparse_path":  sparse_path_1s,
                "index": index,
                "img_name": img_idx,
                "dataset": self.dataset_name}
        if self.pseudo_gt_dir is not None:
            out["gt_S"] = gt_S
        return out

    def __len__(self):
        return len(self.img_list)


np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def custom_collate(batch):  # [numpy array, numpy array] => [torch.Tensor, torch.Tensor]
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * Sequence[NumPy Array] -> Sequence[torch.Tensor]
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        Args:
            batch: a single batch to be collated
        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return [torch.as_tensor(b) for b in batch]
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: custom_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [custom_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([custom_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def is_IIW_complete(dataset_dir, pseudo_gt_dir,  _batch_size, num_workers, visualize_dir=None):
    import time
    import torchvision
    disp_iters = 10

    for mode in ["train", "val", "test"]:
        # visualize
        if visualize_dir is not None:
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir)
            sample_dir = os.path.join(visualize_dir, f"{mode}_samples")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
        # data loader
        target_img_size = None if mode in ["test", "val"] else (224, 224)
        bs = 1 if mode in ["test", "val"] else _batch_size
        dataset_iiw = IIWDataset(dataset_dir, -1, mode,
                                 img_size=target_img_size, pseudo_gt_dir=pseudo_gt_dir)
        data_loader_iiw = torch.utils.data.DataLoader(dataset_iiw,
                                                      batch_size=bs,
                                                      num_workers=num_workers,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      collate_fn=custom_collate)
        print(f"\nIIW_{mode}: \n"
              f"\tsize: {len(dataset_iiw)}, batch_size: {bs}")
        end = time.time()
        for i, data_iiw in enumerate(data_loader_iiw):
            sample = data_iiw["srgb_img"]
            if (i + 1) % disp_iters == 0:
                print(f"\tindex: {i}, image shape: {sample.shape}, "
                      f"\ttime/batch: {(time.time() - end) / disp_iters}")
                end = time.time()
            if sample_dir is not None and ((i+1) % (50//bs) == 0):
                vis_imgs = [v[0] for k, v in data_iiw.items()
                            if k not in ["index", "dataset", "img_name", "targets"]]
                torchvision.utils.save_image(vis_imgs,
                                             os.path.join(sample_dir, f"{i}-0-{data_iiw['img_name'][0]}.jpeg"))
    print("Complete check IIW dataset.")
    