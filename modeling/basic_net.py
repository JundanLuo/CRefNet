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
import torch.nn as nn
from timm.models.layers import trunc_normal_


def g_freeze(model):
    """
    Freeze all params for inference.
    """
    if hasattr(model, "freeze"):
        model.freeze()
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()


def g_unfreeze(model):
    """
    Unfreeze all parameters for training.
    """
    if hasattr(model, "unfreeze"):
        model.unfreeze()
    else:
        for param in model.parameters():
            param.requires_grad = True
        model.train(True)


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

    def freeze(self) -> None:
        """
        Freeze all params for inference.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        """
        for param in self.parameters():
            param.requires_grad = True
        self.train(True)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def configure_params_for_optimizer(self, *args, **kwargs):
        assert False, f"Should be overwritten"


class DataParallelWrapper(nn.Module):
    def __init__(self, model):
        super(DataParallelWrapper, self).__init__()
        self.arch_name = model.arch_name
        self.model = nn.DataParallel(model)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        out["color_rep"] = self.model.module.color_rep
        return out

    def configure_params_for_optimizer(self, *args, **kwargs):
        return self.model.module.configure_params_for_optimizer(*args, **kwargs)

    def unfreeze(self, *args, **kwargs):
        return self.model.module.unfreeze(*args, **kwargs)

    def freeze(self, *args, **kwargs):
        return self.model.module.freeze(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.module.load_state_dict(*args, **kwargs)


class ModuleWrapper(BasicNet):
    def __init__(self, model):
        super(ModuleWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
