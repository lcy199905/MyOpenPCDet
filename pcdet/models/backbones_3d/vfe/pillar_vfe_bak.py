import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from efficientnet_pytorch.model import MemoryEfficientSwish
from einops import rearrange
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inter_channel = channels // 2
        self.global_q = nn.Conv1d(channels, self.inter_channel, 1, bias=True)
        self.global_kv = nn.Conv1d(channels, self.inter_channel * 2, 1, bias=True)
        self.avg_pool = nn.AvgPool1d(1)

        self.local_qkv = nn.Conv1d(channels, self.inter_channel * 3, 1, bias=True)
        self.conv_dw = nn.Conv1d(self.inter_channel, self.inter_channel, 1, groups=self.inter_channel)

        self.fc = nn.Conv1d(self.inter_channel, self.inter_channel, 1)
        self.swish = MemoryEfficientSwish()
        self.tanh = nn.Tanh()

        self.trans_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.befer_norm = nn.GroupNorm(1, channels)
        self.after_norm = nn.GroupNorm(1, channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.fc1 = nn.Conv1d(channels, channels, 1)
        self.dwconv = nn.Conv1d(channels, channels, 1, groups=channels)
        self.fc2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):

        input_data = x
        x = self.befer_norm(x)
        q = self.global_q(x).permute(0, 2, 1) #BNC
        kv = self.global_kv(x).permute(0, 2, 1) #BNC
        global_kv = rearrange(kv, 'b n (m c)->m b n c', m=2)
        global_x_q = q# b, n, c

        global_x_k = global_kv[0].permute(0, 2, 1)  # b, c, n
        global_x_k = self.avg_pool(global_x_k)

        global_x_v = global_kv[1].permute(0, 2, 1)
        global_x_v = self.avg_pool(global_x_v)

        energy = global_x_q @ global_x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = global_x_v @ attention  # b, c, n

        qkv = self.local_qkv(x).permute(0, 2, 1)
        local_qkv = rearrange(qkv, 'b n (m c)->m b n c', m=3)
        local_q = self.conv_dw(local_qkv[0].permute(0, 2, 1))
        local_k = self.conv_dw(local_qkv[1].permute(0, 2, 1))
        local_v = self.conv_dw(local_qkv[2].permute(0, 2, 1))
        attn = torch.mul(local_q, local_k)
        attn = self.fc(self.swish(self.fc(attn)))
        attn = self.tanh(torch.mul(attn, (1 ** -0.5)))
        local_r = torch.mul(attn, local_v)

        x = torch.cat([local_r, x_r], dim=1)
        x = self.trans_conv(x)

        x = input_data + self.act(self.after_norm(x))

        return x
class PFNLayer_SA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True):
        super().__init__()

        self.use_norm = use_norm

        if self.use_norm:
            # 根据论文中，这是是简化版pointnet网络层的初始化
            # 论文中使用的是 1x1 的卷积层完成这里的升维操作（理论上使用卷积的计算速度会更快）
            # 输入的通道数是刚刚经过数据增强过后的点云特征，每个点云有10个特征，
            # 输出的通道数是128
            # self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # 一维BN层
            # self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

            self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, bias=False)
            self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=1, bias=False)

            self.bn1 = nn.BatchNorm1d(out_channels // 4)
            self.bn2 = nn.BatchNorm1d(out_channels // 4)

            self.sa1 = SA_Layer(out_channels // 4)
            self.sa2 = SA_Layer(out_channels // 4)
            self.sa3 = SA_Layer(out_channels // 4)
            self.sa4 = SA_Layer(out_channels // 4)
            self.conv_fuse = nn.Sequential(nn.Conv1d(138, 128, kernel_size=1, bias=False),
                                           nn.BatchNorm1d(128),
                                           nn.LeakyReLU(negative_slope=0.2))
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()
        self.part = 50000

    def forward(self, inputs):

        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        inputs = inputs.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(inputs)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        o = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.cat([o, inputs], dim=1)
        # 138 -> 128
        x = self.conv_fuse(x)
        x = x.permute(0, 2, 1)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                # PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
                PFNLayer_SA(in_filters, out_filters),
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
