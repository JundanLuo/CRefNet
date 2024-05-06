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
import shutil

import torch
from yacs.config import CfgNode as CN


def save_model(model,
               save_dir, label,
               is_best: bool = False,
               other_info: dict = None,
               model_only=False):
    if model_only:
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
    else:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "other_info": other_info,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, label + '.pt')
    torch.save(checkpoint, filepath)
    print('Save checkpoint file: %s' % filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'best_model.pt'))
    return filepath


def load_model(model, file_path, strict=True):
    print('\nLoading model......')

    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))  # Load all tensors onto the CPU
        state_dict = checkpoint.get("model_state_dict", None)
        if state_dict is None:
            state_dict = checkpoint.get("state_dict", None)
        elif checkpoint.get("state_dict", None) is not None:
            raise Exception(f"Checkpoint file has both \'state_dict\' and \'model_state_dict\' keys")
        if state_dict is None:
            state_dict = checkpoint.get("params", None)
        elif checkpoint.get("params", None) is not None:
            raise Exception(f"Checkpoint file has duplicate keys.")
        if state_dict is None:
            print(f"\tno key 'model_state_dict' in the file: {file_path}")
            return False
        else:
            keys = model.load_state_dict(state_dict, strict=strict)
            print(f"\t successfully loaded model from: {file_path}")
            if len(keys.missing_keys) > 0:
                print("\tmissing keys:")
                print("\t\t", keys.missing_keys)
            if len(keys.unexpected_keys) > 0:
                print("\tunexpected_keys")
                print("\t\t", keys.unexpected_keys)
            return True
    else:
        print(" \tno checkpoint found at '{}'. Loading failed!".format(file_path))
        return False
