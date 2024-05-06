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


import math
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from kornia.geometry import resize

from modeling.vqgan.modules.diffusionmodules.model import ResnetBlock, Downsample, Normalize, Upsample, nonlinearity
from modeling.swin_transformer_v2 import BasicLayer as SwinBasicLayer
from modeling.basic_net import BasicNet, g_freeze, g_unfreeze
from utils.image_util import srgb_to_rgb


class PatchEmbed(BasicNet):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.img_size = img_size
        self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if patch_size[0] > 1 or patch_size[1] > 1 or in_chans != embed_dim:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Identity()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # Comment to support different image resolution
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

    # def flops(self):
    #     Ho, Wo = self.patches_resolution
    #     flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
    #     if self.norm is not None:
    #         flops += Ho * Wo * self.embed_dim
    #     return flops


class UpBlock(BasicNet):
    def __init__(self, in_chans, out_chans, scale_factor, conv_last: bool, activate_last: bool):
        super(UpBlock, self).__init__()
        assert conv_last or (not conv_last and in_chans == out_chans)
        if conv_last:
            self.op_last = nn.Sequential(
                nn.Conv2d(out_chans, out_chans, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True) if activate_last else nn.Identity()
            )
        else:
            self.op_last = nn.LeakyReLU(negative_slope=0.2, inplace=True) if activate_last else nn.Identity()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.op_last(x)
        return x


class Encoder(BasicNet):
    def __init__(self, in_channels, ch, ch_mult, num_res_blocks):
        super(Encoder, self).__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.ch_mult = ch_mult
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        resamp_with_conv = True
        dropout = 0.0

        # downsampling
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        self.apply(self._weights_init)

    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        return hs[-1]


class Decoder(BasicNet):
    def __init__(self, in_channels, ch, ch_mult, out_ch, num_res_blocks):
        super(Decoder, self).__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.out_ch = out_ch
        resamp_with_conv = True
        dropout = 0.0

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            # block_out = ch*ch_mult[i_level]
            block_out = ch*in_ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Sequential(
            torch.nn.Conv2d(block_in,
                            ch,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            Normalize(ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ch, out_ch, 1, 1, 0)
        )

        self.apply(self._weights_init)

    def forward(self, z):
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class TransformerModule(BasicNet):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]):
        assert len(num_heads) == len(depths) == len(pretrained_window_sizes)
        super(TransformerModule, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        # Note that projection is not used if patch_size==1 and in_chans==embed_dim
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = tuple(self.patch_embed.patches_resolution)

        assert drop_rate == 0
        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        swin_ft_resolution = (img_size // patch_size, img_size//patch_size)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinBasicLayer(dim=embed_dim,
                                   input_resolution=swin_ft_resolution,
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=None,
                                   use_checkpoint=use_checkpoint,
                                   pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
            self.norms.append(norm_layer(embed_dim))

        # init weights
        self.apply(self._weights_init)
        for bly in self.layers:
            bly._init_respostnorm()

    # def no_weight_decay(self):
    #     return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        # B, _, H, W = x.shape
        x = self.patch_embed(x)
        # x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            B, C, H, W = x.shape
            x = x.view(B, C, -1).transpose(1, 2)
            x = self.norms[i](x)
            x = x.transpose(1, 2).view(B, C, H, W)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class CRefNet(BasicNet):
    arch_name = "CRefNet"
    training_strategy = None

    def __init__(self, in_chans=3, out_r_chans=3, out_s_chans=1, input_img_size=224,
                 num_feat=64, enc_num_res_blocks=2, dec_num_res_blocks=1,
                 num_swin_groups=8, depth_per_swin_group=4,
                 color_rep="rgI",
                 use_checkpoint=True):
        assert isinstance(input_img_size, int), f"input_img_size should be an integer."
        super(CRefNet, self).__init__()
        # encoder
        self.img_size = input_img_size
        self.num_feat = num_feat
        enc_ch_mult = [1, 2, 4]
        # enc_num_res_blocks = 2

        # transformer moduel
        assert self.img_size % (2**(len(enc_ch_mult)-1)) == 0
        in_transformer_size = self.img_size // (2**(len(enc_ch_mult)-1))
        self.patch_size = 1
        embed_dim = self.num_feat * enc_ch_mult[-1]
        num_head = 8
        window_size = 14

        # decoder
        dec_ch_mult = enc_ch_mult
        self.out_r_chans, self.out_s_chans = out_r_chans, out_s_chans
        self.color_rep = color_rep
        # dec_num_res_blocks = 1

        # Reflectance model
        self.r_model = nn.Module()
        self.r_model.encoder_I = Encoder(in_chans, self.num_feat, enc_ch_mult, enc_num_res_blocks)
        self.r_model.encoder_R = Encoder(out_r_chans, self.num_feat, enc_ch_mult, enc_num_res_blocks)
        self.r_model.transf_module = TransformerModule(
            img_size=in_transformer_size, patch_size=self.patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            depths=[depth_per_swin_group]*num_swin_groups, num_heads=[num_head]*num_swin_groups, window_size=window_size,
            mlp_ratio=4., qkv_bias=True,
            norm_layer=nn.LayerNorm, patch_norm=True,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=[0]*num_swin_groups
        )
        self.r_model.decoder_R = Decoder(embed_dim, self.num_feat, dec_ch_mult, out_r_chans, dec_num_res_blocks)

        # Shading model
        self.s_model = nn.Module()
        self.s_model.encoder_I = Encoder(in_chans, self.num_feat, enc_ch_mult, enc_num_res_blocks)
        self.s_model.encoder_S = Encoder(abs(self.out_s_chans), self.num_feat, enc_ch_mult, enc_num_res_blocks)
        self.s_model.transf_module = TransformerModule(
            img_size=in_transformer_size, patch_size=self.patch_size,
            in_chans=embed_dim, embed_dim=embed_dim,
            depths=[depth_per_swin_group]*num_swin_groups, num_heads=[num_head]*num_swin_groups, window_size=window_size,
            mlp_ratio=4., qkv_bias=True,
            norm_layer=nn.LayerNorm, patch_norm=True,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=[0]*num_swin_groups
        )
        self.s_model.decoder_S = Decoder(embed_dim, self.num_feat, dec_ch_mult, abs(self.out_s_chans), dec_num_res_blocks)

        self.min_input_size = (2 ** (len(enc_ch_mult)-1)) * self.patch_size * window_size

    def _generate_reflectance_images(self, x, pred_size):
        if self.color_rep == "rgI":
            out_linear_R = x
            # out_log_R = torch.cat((out_linear_R[:, :2, :, :], torch.log(out_linear_R[:, 2:, :, :].clamp(min=1e-6))), dim=1)
            rg = out_linear_R[:, :2, :, :].clamp(min=0.0, max=1.0)
            b = 1.0 - rg.sum(dim=1, keepdim=True).clamp(max=1.0)
            chromat = torch.cat((rg, b), dim=1)
            pred_R = chromat * out_linear_R[:, 2:, :, :].clamp(min=0.0, max=1.0) * 3
            out = {
                "pred_R_wo_resize": pred_R,
                "direct_linear_out_R": out_linear_R,
                # "direct_log_out_R": out_log_R,
            }
        elif self.color_rep == "RGB":
            out_linear_R = x
            # out_log_R = torch.log(out_linear_R.clamp(min=1e-6))
            pred_R = out_linear_R
            out = {
                "pred_R_wo_resize": pred_R,
                "direct_linear_out_R": out_linear_R,
                # "direct_log_out_R": out_log_R,
            }
        else:
            assert False, f"Not supports color representation: {self.color_rep}"
        if pred_size is not None:
            out["pred_R"] = resize(out["pred_R_wo_resize"], size=pred_size,
                                   interpolation='bilinear', align_corners=False, antialias=True)
        else:
            out["pred_R"] = out["pred_R_wo_resize"]
        return out

    def _generate_shading_images(self, x, pred_size):
        if self.color_rep == "rgI":
            out_linear_S = x
            if abs(self.out_s_chans == 1):
                pred_S = out_linear_S.clamp(min=0.0).repeat(1, 3, 1, 1)
            elif abs(self.out_s_chans == 3):
                rg = out_linear_S[:, :2, :, :].clamp(min=0.0, max=1.0)
                b = 1.0 - rg.sum(dim=1, keepdim=True).clamp(max=1.0)
                chromat = torch.cat((rg, b), dim=1)
                pred_S = chromat * out_linear_S[:, 2:, :, :].clamp(min=0.0) * 3
            else:
                assert False, f"Error number of shading channels: {self.out_s_chans}"
            out = {
                "pred_S_wo_resize": pred_S,
                "direct_linear_out_S": out_linear_S,
            }
        else:
            assert False, f"Not implements color representation: {self.color_rep} for shading"
        if pred_size is not None:
            out["pred_S"] = resize(out["pred_S_wo_resize"], size=pred_size,
                                   interpolation='bilinear', align_corners=False,  antialias=True)
        else:
            out["pred_S"] = out["pred_S_wo_resize"]
        return out

    def resize_input(self, img, resize_img=True, min_dim=None, max_dim=None):
        """
        Resize the input image to the specified size or the self.img_size
        :param img: input image
        :param resize_img: if resize the input image
        :param min_dim: the minimum size of the resized image
        :param max_dim: the maximum size of the resized image
        :return: resized image
        """
        # assert not (min_dim is not None and max_dim is not None), f"min_dim and max_dim cannot be set at the same time."
        if resize_img:
            assert min_dim or max_dim, f"min_dim or max_dim should be set."
        if min_dim is not None:
            assert isinstance(min_dim, int), f"min_dim should be an integer."
            assert min_dim % self.min_input_size == 0, f"min_dim should be a multiple of {self.min_input_size}."
        if max_dim is not None:
            assert isinstance(max_dim, int), f"max_dim should be an integer."
            assert max_dim % self.min_input_size == 0, f"max_dim should be a multiple of {self.min_input_size}."
        if min_dim is not None and max_dim is not None:
            assert min_dim <= max_dim, f"min_dim should be less than or equal to max_dim."
            
        # compute the target size
        h, w = img.shape[2:]
        if resize_img:  # resize the input image to the specified dim
            if min_dim is not None:
                s_min = float(min_dim) / min(h, w)
            else:
                s_min = - 1.0
            if max_dim is not None:
                s_max = float(max_dim) / max(h, w)
            else:
                s_max = - 1.0
            if h < w:
                s_h = s_min if s_min > 0 else s_max
                s_w = s_max if s_max > 0 else s_min
            else:
                s_h = s_max if s_max > 0 else s_min
                s_w = s_min if s_min > 0 else s_max
            del s_min, s_max
        else:  # try to preserve the original input image size
            s_h = s_w = 1.0
        t_h = math.floor(s_h * h + 0.01)
        t_w = math.floor(s_w * w + 0.01)
        t_h = math.floor(t_h / self.min_input_size + 0.01) * self.min_input_size
        t_w = math.floor(t_w / self.min_input_size + 0.01) * self.min_input_size
        if min_dim is not None:
            assert min(t_h, t_w) == min_dim, f"min(t_h, t_w) should be equal to min_dim."
        if max_dim is not None:
            assert max(t_h, t_w) == max_dim, f"max(t_h, t_w) should be equal to max_dim."

        # resize the input image
        if min(h, w) > max(t_h, t_w):
            img = resize(img, size=[t_h, t_w], interpolation="area", antialias=True)
        else:
            img = resize(img, size=[t_h, t_w],
                         interpolation="bilinear", align_corners=False, antialias=True)
        return img

    def forward(self, input_srgb, mode="RE", resize_input=True,
                min_img_size=None, max_img_size=None,
                output_original_size=True):
        """
        :param input_srgb: input image
        :param mode:
            "RR": reflectance reconstruction
            "RE": reflectance estimation
            "SR": shading reconstruction
            "SE": shading estimation
            "IID": intrinsic image decomposition
        :param resize_input:
            if resize the input image or try to preserve the original size
        :param max_img_size:
            if resize the input image to the overwritten max_img_size or the self.img_size
        :param output_original_size:
            if resize the output to the size of the original input image
        :return: predictions
        """
        # rescale input
        x = input_srgb
        pred_size = (x.size(2), x.size(3)) if output_original_size else None
        if isinstance(min_img_size, int) and min_img_size <= 0:
            min_img_size = None
        if isinstance(max_img_size, int) and max_img_size <= 0:
            max_img_size = None
        if max_img_size is None and min_img_size is None and resize_input:
            min_img_size = self.img_size  # default size
        x = self.resize_input(x, resize_input,
                              min_dim=min_img_size,
                              max_dim=max_img_size)
        # print(f"direct input size: {x.shape}")

        # forwarding
        if mode == "RR":  # forward reflectance reconstruction branch only
            x_enc = self.r_model.encoder_R(x)
            x_out = self.r_model.decoder_R(x_enc)
            out = self._generate_reflectance_images(x_out, pred_size)
            out["pred_S"] = torch.ones_like(out["pred_R"])
        elif mode == "RE":  # forward reflectance estimation branch only
            x_enc = self.r_model.encoder_I(x)
            x_former = self.r_model.transf_module(x_enc)
            x_out = self.r_model.decoder_R(x_former)
            out = self._generate_reflectance_images(x_out, pred_size)
            out["pred_S"] = torch.ones_like(out["pred_R"])
        elif mode == "SE":  # forward shading estimation branch only
            x_enc = self.s_model.encoder_I(x)
            x_former = self.s_model.transf_module(x_enc)
            x_out = self.s_model.decoder_S(x_former)
            out = self._generate_shading_images(x_out, pred_size)
            out["pred_R"] = torch.ones_like(out["pred_S"])
        elif mode == "SR_share_BN":  # forward SE branch and SR branch jointly
            in_img, in_S = torch.split(x, x.size(0)//2)
            if abs(self.out_s_chans) == 1:
                in_S = in_S.mean(dim=1, keepdim=True)
            x_enc_img = self.s_model.encoder_I(in_img)
            x_former = self.s_model.transf_module(x_enc_img)
            x_enc_S = self.s_model.encoder_S(in_S)
            x_ = torch.cat([x_former, x_enc_S], dim=0)
            x_out = self.s_model.decoder_S(x_)
            out = self._generate_shading_images(x_out, pred_size)
            out["pred_R"] = torch.ones_like(out["pred_S"])
        elif mode == "SR":  # forward shading reconstruction branch only
            if abs(self.out_s_chans) == 1:  # grayscale shading
                in_s = x.mean(dim=1, keepdim=True)
            else:
                in_s = x
            x_enc_s = self.s_model.encoder_S(in_s)
            x_out = self.s_model.decoder_S(x_enc_s)
            out = self._generate_shading_images(x_out, pred_size)
            out["pred_R"] = torch.ones_like(out["pred_S"])
        elif mode == "IID":
            x_enc = self.r_model.encoder_I(x)
            x_former = self.r_model.transf_module(x_enc)
            x_out = self.r_model.decoder_R(x_former)
            out = self._generate_reflectance_images(x_out, pred_size)
            if self.out_s_chans <= 0:  # indicates shading has not been trained. compute shading by dividing reflectance
                input_rgb = srgb_to_rgb(input_srgb)
                mask = (torch.min(out["pred_R"], dim=1, keepdim=True)[0] > 1e-3).to(torch.float32).repeat(1, 3, 1, 1)
                pred_s = (input_rgb / out["pred_R"].clamp(min=1e-6))*mask
                out["pred_S"] = pred_s.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            else:
                _x_enc = self.s_model.encoder_I(x)
                _x_former = self.s_model.transf_module(_x_enc)
                _x_out = self.s_model.decoder_S(_x_former)
                _out = self._generate_shading_images(_x_out, pred_size)
                out = {**out, **_out}
        else:
            raise Exception(f"Not support mode: {mode}")
        out["color_rep"] = self.color_rep
        return out

    def configure_params_for_optimizer(self, cfg):
        if cfg.TRAIN.STRATEGY.type == "RE":
            optim_params = [{'params': chain(self.r_model.encoder_I.parameters(),
                                             self.r_model.transf_module.parameters(),
                                             self.r_model.decoder_R.parameters()),
                             'lr': cfg.TRAIN.lr,
                             'name': 'reflectance estimation'}]
        elif cfg.TRAIN.STRATEGY.type == "with_RR":  # reflectance map reconstruction
            optim_params = [{'params': chain(self.r_model.parameters()),
                             'lr': cfg.TRAIN.lr,
                             'name': 'with reflectance reconstruction'}]
        elif cfg.TRAIN.STRATEGY.type == "SE":
            optim_params = [{'params': chain(self.s_model.encoder_I.parameters(),
                                             self.s_model.transf_module.parameters(),
                                             self.s_model.decoder_S.parameters()),
                             'lr': cfg.TRAIN.lr,
                             'name': 'shading estimation'}]
        elif cfg.TRAIN.STRATEGY.type == "with_SR":  # shading map reconstruction
            optim_params = [{'params': self.s_model.parameters(),
                             'lr': cfg.TRAIN.lr,
                             'name': 'with shading reconstruction'}]
        else:
            assert False, f"Not supports training strategy {cfg.TRAIN.STRATEGY.type}"
        self.training_strategy = cfg.TRAIN.STRATEGY.type
        return optim_params

    def unfreeze(self) -> None:
        """
        Unfreeze parameters for training according to the training strategy.
        """
        if self.training_strategy == "RE":
            g_unfreeze(self.r_model)
            g_freeze(self.s_model)
            self.r_model.encoder_R.freeze()
        elif self.training_strategy == "with_RR":  # reflectance map reconstruction
            g_unfreeze(self.r_model)
            g_freeze(self.s_model)
        elif self.training_strategy == "SE":
            g_freeze(self.r_model)
            g_unfreeze(self.s_model)
            self.s_model.encoder_S.freeze()
        elif self.training_strategy == "with_SR":  # shading map reconstruction
            g_freeze(self.r_model)
            g_unfreeze(self.s_model)
        else:
            raise Exception(f"Undefined training strategy: {self.training_strategy}")
