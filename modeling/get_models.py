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


def get(cfg):
    architecture = cfg.MODEL.ARCH.name
    r_chs, s_chs = cfg.MODEL.ARCH.r_chs, cfg.MODEL.ARCH.s_chs
    num_swin_groups = cfg.MODEL.ARCH.num_swin_groups
    depth_per_swin_group = cfg.MODEL.ARCH.depth_per_swin_group
    pretrained_file = cfg.TRAIN.pretrained
    input_img_size = cfg.MODEL.input_img_size
    color_rep = cfg.MODEL.color_rep
    use_checkpoint_tool = cfg.TRAIN.use_checkpoint_tool
    if architecture == "crefnet":
        from modeling.crefnet_swin_v2 import CRefNet
        model = CRefNet(3, r_chs, s_chs, input_img_size,
                        num_swin_groups=num_swin_groups, depth_per_swin_group=depth_per_swin_group,
                        color_rep=color_rep,
                        use_checkpoint=use_checkpoint_tool)
    elif architecture == "crefnet-e":
        from modeling.crefnet_swin_v2 import CRefNet
        assert num_swin_groups == 4 and depth_per_swin_group == 4, \
            "crefnet-e should have num_swin_groups=4 and depth_per_swin_group=4"
        model = CRefNet(3, r_chs, s_chs, input_img_size,
                        num_feat=32, enc_num_res_blocks=1,
                        num_swin_groups=4, depth_per_swin_group=4,
                        color_rep=color_rep,
                        use_checkpoint=use_checkpoint_tool)
    else:
        raise Exception(f"Not support model architecture: {architecture}")
    print(f"\nBuild Model {architecture}:{model.arch_name} with color_rep {color_rep}")
    return model
