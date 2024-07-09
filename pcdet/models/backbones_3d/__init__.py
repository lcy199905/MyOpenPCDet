from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelResBackBone8x_V2
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x, PillarRes34BackBone8x, PillarBackBone8x_origin, PillarRes18BackBone8xM
# from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
from .spconv_unet import UNetV2
from .dsvt import DSVT

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelResBackBone8x_V2': VoxelResBackBone8x_V2,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarBackBone8x_origin': PillarBackBone8x_origin,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'PillarRes34BackBone8x': PillarRes34BackBone8x,
    'DSVT': DSVT,
    'PillarRes18BackBone8xM': PillarRes18BackBone8xM
}
