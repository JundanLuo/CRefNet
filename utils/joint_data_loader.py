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


import random

import torch.utils.data as data


class JointDataLoader(object):
    last_idx = -1
    get_next_idx = None

    def __init__(self, data_loaders: list = [], names: list = [], mode="sequence"):
        self.data_loaders = data_loaders
        self.names = names
        self.iterators = []
        for dl in self.data_loaders:
            self.iterators.append(iter(dl))
        self._reset_selection_mode(mode)

    def _reset_selection_mode(self, mode=None):
        self.last_idx = -1
        if mode is not None:
            self.get_next_idx = {
                "sequence": self._get_sequence_idx,
                "random": self._get_random_idx
            }[mode]

    def append(self, data_loader, name: str):
        self.data_loaders.append(data_loader)
        self.names.append(name)
        self.iterators.append(iter(self.data_loaders[-1]))
        return self

    def concat(self, y, mode):
        assert len(y.data_loaders) == len(y.iterators) == len(y.names)
        self.data_loaders += y.data_loaders
        self.names += y.names
        self.iterators += y.iterators
        self._reset_selection_mode(mode)
        return self

    def _remove_by_index(self, idx):
        self.data_loaders.pop(idx)
        self.names.pop(idx)
        self.iterators.pop(idx)
        self._reset_selection_mode()
        return self

    def remove(self, name):
        flag = 0
        find_idx = None
        for i in range(len(self.names)):
            if self.names[i] == name:
                find_idx = i
                flag += 1
        assert flag == 1, f"Find name:{name} {flag} times!"
        self._remove_by_index(find_idx)
        return self

    def _get_sequence_idx(self):
        idx = self.last_idx + 1
        if idx >= len(self.iterators):
            idx = 0
        return idx

    def _get_random_idx(self):
        return random.randint(0, len(self.iterators)-1)

    def __len__(self):
        return len(self.data_loaders)

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.get_next_idx()
        try:
            data = next(self.iterators[idx])  # Samples the batch
        except StopIteration:
            self.iterators[idx] = iter(self.data_loaders[idx])  # restart if the previous generator is exhausted
            data = next(self.iterators[idx])
        self.last_idx = idx
        return data


def merge_datasets(dataset_x: data.dataset, dataset_y: data.dataset, balanced: bool = True):
    """
    Merge two datasets
    :param dataset_x: one dataset
    :param dataset_y: another dataset
    :param balanced: if True, the shorter dataset will be repeated to match the length of the longer one
    :return: a ConcatDataset
    """
    assert len(dataset_x) > 0 and len(dataset_y) > 0
    if balanced:
        length = max(len(dataset_x), len(dataset_y))
        if len(dataset_x) < length:
            dataset_x = data.ConcatDataset([dataset_x] * (length // len(dataset_x)))
        elif len(dataset_y) < length:
            dataset_y = data.ConcatDataset([dataset_y] * (length // len(dataset_y)))
        assert len(dataset_x) > 0 and len(dataset_y) > 0
    return data.ConcatDataset([dataset_x, dataset_y])


