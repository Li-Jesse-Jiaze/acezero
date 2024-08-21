# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


# class Encoder(nn.Module):
#     """
#     FCN encoder, used to extract features from the input images.

#     The number of output channels is configurable, the default used in the paper is 512.
#     """

#     def __init__(self, out_channels=512):
#         super(Encoder, self).__init__()

#         self.out_channels = out_channels

#         self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
#         self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

#         self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
#         self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

#         self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
#         self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

#         self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         res = F.relu(self.conv4(x))

#         x = F.relu(self.res1_conv1(res))
#         x = F.relu(self.res1_conv2(x))
#         x = F.relu(self.res1_conv3(x))

#         res = res + x

#         x = F.relu(self.res2_conv1(res))
#         x = F.relu(self.res2_conv2(x))
#         x = F.relu(self.res2_conv3(x))

#         x = self.res2_skip(res) + x

#         return x

class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    """
    Implementation of architecture described in
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4, ceil_mode=True),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0),
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws=2):
        """
        Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        """
        input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
        return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        # dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        feats = self.block_fusion(x3 + x4 + x5)

        # heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits)
        scores = F.softmax(keypoints, 1)[:, :64]
        position = torch.argmax(scores, dim=1).squeeze(1)  # [b, h, w] position
        rows, columns = torch.div(position, 8, rounding_mode='floor'), torch.fmod(position, 8)
        positions = torch.stack((rows, columns), dim=1)

        # return feats, keypoints, heatmap
        return feats, positions, heatmap


class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(
        self,
        mean,
        num_head_blocks,
        use_homogeneous,
        homogeneous_min_scale=0.01,
        homogeneous_max_scale=4.0,
        in_channels=512,
    ):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = (
            nn.Identity()
            if self.in_channels == self.head_channels
            else nn.Linear(self.in_channels, self.head_channels)
        )

        self.res3_linear1 = nn.Linear(self.in_channels, self.head_channels)
        self.res3_linear2 = nn.Linear(self.head_channels, self.head_channels)
        self.res3_linear3 = nn.Linear(self.head_channels, self.head_channels)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append(
                (
                    nn.Linear(self.head_channels, self.head_channels),
                    nn.Linear(self.head_channels, self.head_channels),
                    nn.Linear(self.head_channels, self.head_channels),
                )
            )

            super(Head, self).add_module(str(block) + "l0", self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + "l1", self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + "l2", self.res_blocks[block][2])

        self.fc1 = nn.Linear(self.head_channels, self.head_channels)
        self.fc2 = nn.Linear(self.head_channels, self.head_channels)

        if self.use_homogeneous:
            self.fc3 = nn.Linear(self.head_channels, 4)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1.0 / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1.0 - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1.0 / self.min_scale)
        else:
            self.fc3 = nn.Linear(self.head_channels, 3)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3))

    def forward(self, res):
        x = F.relu(self.res3_linear1(res))
        x = F.relu(self.res3_linear2(x))
        x = F.relu(self.res3_linear3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            # Dehomogenize coords:
            h_slice = (
                F.softplus(sc[:, 3].unsqueeze(-1), beta=self.h_beta.item())
                + self.max_inv_scale
            )
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc


class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(
        self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512
    ):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder()
        self.heads = Head(
            mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim
        )

    @classmethod
    def create_from_encoder(
        cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous
    ):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = 64

        # Create a regressor.
        _logger.info(
            f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size."
        )
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+l0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = 64

        # Create a regressor.
        _logger.info(
            f"Creating regressor from pretrained state_dict:"
            f"\n\tNum head blocks: {num_head_blocks}"
            f"\n\tHomogeneous coordinates: {use_homogeneous}"
            f"\n\tEncoder feature size: {num_encoder_features}"
        )
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        """
        Forward pass.
        """
        # TODO: add px potision
        features, _, _ = self.get_features(inputs)
        b, c, h, w = features.shape
        features = features.view(b, c, -1).permute(0, 2, 1).contiguous().view(-1, c)
        sc = self.get_scene_coordinates(features)
        return sc.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
