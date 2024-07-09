import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class PFNLayerV2(nn.Module):
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

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        x_global = x_max

        if self.last_vfe:
            return x_global
        else:
            x_concatenated = torch.cat([x, x_global[unq_inv, :]], dim=1)
            return x_concatenated

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None


    def forward(self, x):
        x = self.conv(x)
        if hasattr(self,'bn'):
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialGate, self).__init__()
        kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class ZBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, kernel_size=7):
        super(ZBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(kernel_size)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class bin_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, in_channels//2, bias=False),
            nn.BatchNorm1d(in_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class conv1d(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ZcCBAM(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        self.in_channels = model_cfg.input_channel
        self.out_channels = model_cfg.output_channel
        self.num_bins = model_cfg.num_bins
        self.zbam = ZBAM(model_cfg.output_channel)
        self.bin_shuffle = bin_shuffle((self.in_channels)*model_cfg.num_bins, model_cfg.output_channel)
        self.up_dimension = conv1d(input_dim = 4, hidden_dim = int(model_cfg.output_channel/2), output_dim = model_cfg.output_channel, num_layers = 2)
        self.residual = model_cfg.get("z_residual", False)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2
        if 'mask_position' in data_dict:
            occupied_mask = data_dict['mask_position']
            occupied_mask = occupied_mask[data_dict['zfilter']]
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:,:,:4]
        src = src[:,:,1:]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        data_dict['mlp_feat'] = src.permute(0,2,1).contiguous()
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        if self.residual:
            pfe_src = data_dict['x_mean'][occupied_mask]
            src = torch.cat([src, pfe_src],dim=-1)
        return src, occupied_mask, None, None

class HCBAM(ZcCBAM):
    def __init__(self,
                 model_cfg):
        super().__init__(model_cfg=model_cfg)
        self.feature_fusion = model_cfg.get("feature_fusion", 'sum')
        self.bin_shuffle_t = bin_shuffle((4) * 10, model_cfg.output_channel)
        self.residual = model_cfg.get("z_residual", False)

    def binning_t(self, data_dict):
        voxels, voxel_coords = data_dict['toxel_features'], data_dict['toxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['t_feat_unq_coords'], data_dict['t_feat_unq_inv'], data_dict[
            't_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], 10, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:, 0].float()
        occupied_mask = unq_cnt >= 2
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src_t, occupied_mask_t = self.binning_t(data_dict)
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:, :, :4]
        src = src[..., 1:]
        src = self.up_dimension(src)
        src = src.permute(0, 2, 1).contiguous()
        src = self.zbam(src)
        data_dict['mlp_feat'] = src.permute(0, 2, 1).contiguous()
        N, C, Z = src.shape
        src = src.view(N, Z * C)
        src = self.bin_shuffle(src)
        src_t = src_t[occupied_mask_t]
        data_dict['mlp_bxyt'] = src_t[:, :, :4]
        src_t = src_t[..., 1:].contiguous()
        data_dict['mlp_feat_t'] = src_t
        N, T, C = src_t.shape
        src_t = src_t.view(N, T * C)
        src_t = self.bin_shuffle_t(src_t)
        return src, occupied_mask, src_t, occupied_mask_t

def build_mlp(model_cfg, model_name='ZCONV'):
    model_dict = {
        'HCBAM': HCBAM
}
    model_class = model_dict[model_name]

    model = model_class(model_cfg
                        )
    return model


class DSP(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        num_point_features = model_cfg.num_point_features
        self.point_cloud_range = model_cfg.get("point_cloud_range")
        self.voxel_size = model_cfg.get("voxel_size")
        self.grid_size = model_cfg.get("grid_size")
        self.x_offset = model_cfg.get("x_offset")
        self.y_offset = model_cfg.get("y_offset")
        self.z_offset = model_cfg.get("z_offset")
        self.voxel_x = model_cfg.get("voxel_x")
        self.voxel_y = model_cfg.get("voxel_y")
        self.consecutive = model_cfg.get("consecutive", False)
        self.parallel = model_cfg.get("parallel", False)
        self.z_unlimit = model_cfg.get("z_unlimit")
        self.use_cluster_xyz = model_cfg.get("use_cluster_xyz")
        self.voxel_channel = model_cfg.get("voxel_channel", False)
        self.residual = model_cfg.get("residual", False)
        self.reidx = model_cfg.get("reidx", False)
        channel_list = [[num_point_features, 32], \
                        [num_point_features + 32 * 2, self.voxel_channel], \
                        [num_point_features + self.voxel_channel * 2, self.voxel_channel], \
                        [self.voxel_channel * 2, 32]]
        if self.consecutive:
            self.blocks = nn.ModuleList()
            for module_idx in range(len(self.consecutive) + 1):
                if module_idx == len(self.consecutive):
                    module_idx = -1
                input_features = channel_list[module_idx][0]
                output_features = channel_list[module_idx][1]
                self.blocks.append(nn.Sequential(
                    nn.Linear(input_features, output_features, bias=False),
                    nn.BatchNorm1d(output_features, eps=1e-3, momentum=0.01),
                    nn.ReLU()))
        elif self.parallel:
            self.blocks = nn.ModuleList()
            for module_idx in range(2):
                output_features = self.voxel_channel
                input_features = num_point_features if module_idx == 0 else output_features * 2 * len(self.parallel)
                self.blocks.append(nn.Sequential(
                    nn.Linear(input_features, output_features, bias=False),
                    nn.BatchNorm1d(output_features, eps=1e-3, momentum=0.01),
                    nn.ReLU()))

    def forward(self, batch_dict, f_center, unq_inv, f_cluster=None, f_relative=None):
        points = batch_dict['points']
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        if self.consecutive:
            features = None
            for module_idx, module in enumerate(self.consecutive):
                func = getattr(self, module)
                if module == 'standard':
                    features = func(points, f_center, f_cluster, f_relative, unq_inv, features, module_idx)
                else:
                    features = func(points, points_coords_3d, features, batch_dict, module_idx)
                    if self.residual and module_idx == self.residual[0]:
                        residual_feature = features
                    if self.residual and module_idx == self.residual[1]:
                        features = torch.add(residual_feature, features)
            final_features = features
        if self.parallel:
            final_features = []
            for module_idx, module in enumerate(self.parallel):
                func = getattr(self, module)
                if module == 'standard':
                    final_features.append(func(points, f_center, f_cluster, f_relative, unq_inv, None, 0, batch_dict))
                else:
                    final_features.append(func(points, points_coords_3d, None, batch_dict, 0))
            final_features = torch.cat(final_features, dim=1)
        output = self.blocks[-1](final_features)
        batch_dict['dsp_feat'] = output
        output = torch_scatter.scatter_max(output, unq_inv, dim=0)[0]
        return output

    def gen_feat(self, points, f_center, f_cluster, f_relative, unq_inv, batch_dict, prev_features=None, idx=False,
                 type=None):
        features = [f_center]
        features.append(points[:, 1:])
        features.append(f_cluster)
        features.append(f_relative)
        features = torch.cat(features, dim=-1).contiguous()
        features = self.blocks[idx](features)
        x_mean = torch_scatter.scatter_mean(features, unq_inv, dim=0)
        if type == 'standard':
            batch_dict['x_mean'] = x_mean
        features = torch.cat([features, x_mean[unq_inv, :]], dim=1)
        return features

    def upsample(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        downsample_level = 1 / 2
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        if self.z_unlimit:
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]])).all(dim=1)
        else:
            points_z = points_coords_3d[:, -1:]
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]]) & (points_z >= 0) & (
                        points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                       points_coords[:, 0] * scale_y + \
                       points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
        else:
            f_cluster = None
        features = self.gen_feat(points, f_center, f_cluster, unq_inv, features, module_idx)
        return features

    def downsample(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        downsample_level = 2
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        if self.z_unlimit:
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]])).all(dim=1)
        else:
            points_z = points_coords_3d[:, -1:]
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]]) & (points_z >= 0) & (
                        points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                       points_coords[:, 0] * scale_y + \
                       points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
        else:
            f_cluster = None
        features = self.gen_feat(points, f_center, f_cluster, unq_inv, features, module_idx)
        return features

    def shift(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        if 'shift_data' not in batch_dict:
            shifted_point_cloud_range = self.point_cloud_range[[0, 1]] + self.voxel_size[[0, 1]] / 2
            points_coords = (torch.floor(
                (points[:, [1, 2]] - shifted_point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]) + 1).int()
            # points_z = points_coords_3d[:,-1:]
            # mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
            mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1)).all(dim=1)
            points = points[mask]
            points_coords = points_coords[mask]
            points_xyz = points[:, [1, 2, 3]].contiguous()
            shifted_scale_xy = (self.grid_size[0] + 1) * (self.grid_size[1] + 1)
            shifted_scale_y = (self.grid_size[1] + 1)
            merge_coords = points[:, 0].int() * shifted_scale_xy + \
                           points_coords[:, 0] * shifted_scale_y + \
                           points_coords[:, 1]

            _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

            f_center = torch.zeros_like(points_xyz)

            f_center[:, 0] = points_xyz[:, 0] - (
                        points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
            f_center[:, 1] = points_xyz[:, 1] - (
                        points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
            f_center[:, 2] = points_xyz[:, 2] - self.z_offset

            if self.use_cluster_xyz:
                points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
                f_cluster = points_xyz - points_mean[unq_inv, :]
            else:
                f_cluster = None
            f_relative = points_xyz - torch.cat([shifted_point_cloud_range, self.point_cloud_range[2:3]], dim=0)
            batch_dict['shift_data'] = [points, f_center, f_cluster, unq_inv]
        else:
            points, f_center, f_cluster, unq_inv = batch_dict['shift_data']
        features = self.gen_feat(points, f_center, f_cluster, f_relative, unq_inv, batch_dict, features, module_idx)
        return features

    def standard(self, points, f_center, f_cluster, f_relative, unq_inv, features, module_idx, batch_dict):
        features = self.gen_feat(points, f_center, f_cluster, f_relative, unq_inv, batch_dict, features, module_idx,
                                 'standard')
        return features

    def to_dense_batch(self, x, pillar_idx, max_points, max_pillar_idx):
        r"""
        Point sampling according to pillar index with constraint amount
        """

        # num_points in pillars (0 for empty pillar)
        num_nodes = torch_scatter.scatter_add(pillar_idx.new_ones(x.size(0)), pillar_idx, dim=0,
                                              dim_size=max_pillar_idx)
        cum_nodes = torch.cat([pillar_idx.new_zeros(1), num_nodes.cumsum(dim=0)])

        # check if num_points in pillars exceed the predefined num_points value
        filter_nodes = False
        if num_nodes.max() > max_points:
            filter_nodes = True
        tmp = torch.arange(pillar_idx.size(0), device=x.device) - cum_nodes[pillar_idx]
        if filter_nodes:
            mask = tmp < max_points
            x = x[mask]
            pillar_idx = pillar_idx[mask]
        return x, pillar_idx

def build_dsp(model_cfg, model_name='DSP'):
    model_dict = {
        'DSP': DSP
}
    model_class = model_dict[model_name]

    model = model_class(model_cfg
                        )
    return model

class SENet(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(SENet, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=round(in_channel/reduction_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(in_channel/reduction_ratio), out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x[None,:]
        out = self.globalAvgPool(x.permute(0,2,1))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(1, 1, out.size(1))
        out = out * x
        return out.squeeze(0).contiguous()

class FineGrainedPFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', False)
        self.use_relative_xyz = self.model_cfg.get('USE_RELATIVE_XYZ', True)
        if self.use_relative_xyz:
            num_point_features += 3
        if self.use_absolute_xyz:
            num_point_features += 3
        if self.use_cluster_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1
        self.num_point_features = num_point_features
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.zpillar = model_cfg.get("ZPILLAR", None)
        self.numbins = int(8 / voxel_size[2])
        model_cfg.ZPILLAR_CFG.update({"num_bins": self.numbins})
        self.zpillar_model = build_mlp(model_cfg.ZPILLAR_CFG, model_name=self.zpillar)
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        # self.dsp = model_cfg.get("DSP", None)
        # model_cfg.DSP_CFG.update({"num_point_features": num_point_features, "use_cluster_xyz": self.use_cluster_xyz, \
        #                           "point_cloud_range": self.point_cloud_range, "voxel_size": self.voxel_size, \
        #                           "grid_size": self.grid_size, "x_offset": self.x_offset, "y_offset": self.y_offset,
        #                           "z_offset": self.z_offset, \
        #                           "voxel_x": self.voxel_x, "voxel_y": self.voxel_y})
        # self.dsp_model = build_dsp(model_cfg.DSP_CFG, model_name=self.dsp)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.point_cloud_range_t = torch.tensor([[0.01]]).cuda()
        self.temp_size = torch.tensor([[0.05]]).cuda()
        self.grid_t = torch.tensor(10).cuda()
        self.scale_xyt = grid_size[0] * grid_size[1] * self.grid_t
        self.scale_yt = grid_size[1] * self.grid_t
        self.scale_t = self.grid_t
        self.mlp = nn.Sequential(
            nn.Linear(96, 32, bias=False),
            nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
            nn.ReLU())
        self.senet = SENet(96, 8)
        self.double_flip = model_cfg.get("DOUBLE_FLIP", False)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def dyn_voxelization(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       point_coords[:, 0] * self.scale_yz + \
                       point_coords[:, 1] * self.scale_z + \
                       point_coords[:, 2]
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['v_unq_inv'] = unq_inv
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict

    def dyn_voxelization_t(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyt + \
                       point_coords[:, 0] * self.scale_yt + \
                       point_coords[:, 1] * self.scale_t + \
                       point_coords[:, 2]
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyt,
                                    (unq_coords % self.scale_xyt) // self.scale_yt,
                                    (unq_coords % self.scale_yt) // self.scale_t,
                                    unq_coords % self.scale_t), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['t_unq_inv'] = unq_inv
        batch_dict['toxel_features'] = points_mean.contiguous()
        batch_dict['toxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        if self.double_flip:
            batch_dict['batch_size'] = batch_dict['batch_size'] * 4
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        points_coords = points_coords_3d[:, :2]
        points_coords_t = torch.round((points[:, [4]] - self.point_cloud_range_t) / self.temp_size).int()
        points_coords_t[points_coords_t[:, -1] > 9] = 9
        points_coords_t = torch.cat([points_coords, points_coords_t], dim=-1)
        mask3d = ((points_coords_3d >= 0) & (points_coords_3d < self.grid_size)).all(dim=1)
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        zfilter = (mask3d == mask)
        points_coords_3d = points_coords_3d[mask3d]
        points3d = points[mask3d]
        points_coords_t = points_coords_t[mask]
        points = points[mask]
        points_coords = points_coords[mask]

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        points_xyz = points[:, [1, 2, 3]].contiguous()
        zfilter = torch_scatter.scatter_max(zfilter.int(), unq_inv, dim=0)[0]
        batch_dict['zfilter'] = zfilter = zfilter.bool()
        batch_dict['points'] = points
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            features.append(f_cluster)
        else:
            f_cluster = None

        if self.use_relative_xyz:
            f_relative = points_xyz - self.point_cloud_range[:3]
            features.append(f_relative)
        features = torch.cat(features, dim=-1)

        # features = self.dsp_model(batch_dict, f_center, unq_inv, f_cluster, f_relative)
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        batch_dict = self.dyn_voxelization(points3d, points_coords_3d, batch_dict)
        batch_dict = self.dyn_voxelization_t(points, points_coords_t, batch_dict)
        batch_dict['pillar_merge_coords'] = merge_coords
        batch_dict['unq_inv'] = unq_inv
        batch_dict['point_cloud_range'] = self.point_cloud_range
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['grid_size'] = self.grid_size
        voxel_features, voxel_features_coords = batch_dict['voxel_features'], batch_dict['voxel_features_coords']
        v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:,
                                                                      3] * self.scale_y + voxel_features_coords[:, 2]
        v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True,
                                                                         return_counts=True, dim=0)
        batch_dict['v_feat_unq_coords'] = v_feat_unq_coords
        batch_dict['v_feat_unq_inv'] = v_feat_unq_inv
        batch_dict['voxel_features'] = voxel_features
        batch_dict['v_feat_unq_cnt'] = v_feat_unq_cnt
        toxel_features, toxel_features_coords = batch_dict['toxel_features'], batch_dict['toxel_features_coords']
        # V
        t_feat_coords = toxel_features_coords[:, 0] * self.scale_xy + toxel_features_coords[:,
                                                                      3] * self.scale_y + toxel_features_coords[:, 2]
        t_feat_unq_coords, t_feat_unq_inv, t_feat_unq_cnt = torch.unique(t_feat_coords, return_inverse=True,
                                                                         return_counts=True, dim=0)
        batch_dict['t_feat_unq_coords'] = t_feat_unq_coords
        batch_dict['t_feat_unq_inv'] = t_feat_unq_inv
        batch_dict['toxel_features'] = toxel_features
        batch_dict['t_feat_unq_cnt'] = t_feat_unq_cnt
        # T
        z_pillar_feat, occupied_mask, t_pillar_feat, occupied_mask_t = self.zpillar_model(batch_dict)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                    (unq_coords * 0),
                                    (unq_coords % self.scale_xy) // self.scale_y,
                                    unq_coords % self.scale_y,
                                    ), dim=1)
        voxel_coords = voxel_coords[:, [0, 1, 3, 2]]
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        p_num = features.shape[0]
        t_feat = t_pillar_feat.new_zeros((p_num, t_pillar_feat.shape[1]))
        t_feat[occupied_mask_t] = t_pillar_feat
        p_feat = z_pillar_feat.new_zeros((p_num, z_pillar_feat.shape[1]))
        z_feat = p_feat[zfilter]
        z_feat[occupied_mask] = z_pillar_feat
        p_feat[zfilter] = z_feat

        features = torch.cat([features, p_feat, t_feat], dim=1)
        residual = features
        features = self.senet(features)
        features += residual
        features = self.mlp(features)
        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict